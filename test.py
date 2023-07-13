import os.path

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
import PIL.ImageOps
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F
import torchvision.models as models


class Config():
    # 基于两套数据集、两种模型训练下的参数
    __ATT = "ATT"  # ATT训练集拥有40个不同的人的人脸数据集，每个人有5张不同角度的照片，40x5x(100x100)
    __CASIA = "CASIA"  # ATT训练集拥有500个不同的人的人脸数据集，每个人有5张不同角度的照片，500x5x(640x480)
    __LeNet = "[LeNet]"  # LeNet-5模型
    __ResNet = "[ResNet]"  # ResNet-18模型

    # 两种损失函数下的模型参数(triplet loss又分为两种数据采集方式)
    __params_contra = "params_contrastive.pkl"
    __params_triplet_batch_all = "params_triplet_batch_all.pkl"
    __params_triplet_batch_hard = "params_triplet_batch_hard.pkl"

    # 定义训练所使用的网络结构及所用数据集
    model = __LeNet
    data_sets = __ATT
    # 加载测试集路径
    testing_dir = os.path.join("datasets", data_sets, "train")
    # 提供两个模型作为对比
    params_root1 = os.path.join("params", data_sets, model + __params_contra)
    params_root2 = os.path.join("params", data_sets, model + __params_triplet_batch_all)
    params_root3 = os.path.join("params", data_sets, model + __params_triplet_batch_hard)

    test_batch_size = 1000
    # 定义测试集大小和阈值超参数
    if data_sets == __ATT:
        threshold_Contra = 0.85
        threshold_Triplet = 82.
    elif data_sets == __CASIA:
        threshold_Contra = 0.24
        threshold_Triplet = 1.4


def imshow(img, text=None, should_save=False):
    npimg = img.cpu().numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class PairDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.data_set = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.data_set.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.data_set.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.data_set.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # 转换为灰度图
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.data_set)


class TripletDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.data_set = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        path, label = self.data_set[index]
        img = Image.open(path)
        # img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_set)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output


def evaluate(net_Contra, net_Triplet_All, net_Triplet_Hard, test_dataloader):
    ground_truth = []
    predict_Contra = list()
    predict_Triplet_All = list()
    predict_Triplet_Hard = list()

    for i in range(Config.test_batch_size):
        # 随机产生一组图片对，若img0和img1为相同类别，则label=1
        data = next(iter(test_dataloader))
        img0, img1, label = data
        img0 = img0.cuda()
        img1 = img1.cuda()
        label = label.cuda()

        ground_truth.append(torch.squeeze(label.int()).bool())

        # 通过net_Contra计算输出
        embeddings0_Contra = net_Contra(img0)
        embeddings1_Contra = net_Contra(img1)
        euclidean_distance_Contra = F.pairwise_distance(embeddings0_Contra, embeddings1_Contra)
        Dissimilarity_Contra = euclidean_distance_Contra.item()
        predict_Contra.append(Dissimilarity_Contra <= Config.threshold_Contra)
        concatenated = torch.cat((img0, img1), 0)
        # imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(Dissimilarity_Contra))

        # 通过net_Triplet_All计算输出
        embeddings0_Triplet_All = net_Triplet_All(img0)
        embeddings1_Triplet_All = net_Triplet_All(img1)
        euclidean_distance_Triplet = F.pairwise_distance(embeddings0_Triplet_All, embeddings1_Triplet_All)
        Dissimilarity_Triplet = euclidean_distance_Triplet.item()
        predict_Triplet_All.append(Dissimilarity_Triplet <= Config.threshold_Triplet)
        concatenated = torch.cat((img0, img1), 0)
        similarity_percentage = Config.threshold_Triplet / (Config.threshold_Triplet + Dissimilarity_Triplet)
        imshow(torchvision.utils.make_grid(concatenated), 'similarity: {:.2f}%'.format(similarity_percentage * 100))

        # 通过net_Triplet_Hard计算输出
        embeddings0_Triplet_Hard = net_Triplet_Hard(img0)
        embeddings1_Triplet_Hard = net_Triplet_Hard(img1)
        euclidean_distance_Triplet = F.pairwise_distance(embeddings0_Triplet_Hard, embeddings1_Triplet_Hard)
        Dissimilarity_Triplet = euclidean_distance_Triplet.item()
        predict_Triplet_Hard.append(Dissimilarity_Triplet <= Config.threshold_Triplet)
        concatenated = torch.cat((img0, img1), 0)
        similarity_percentage = Config.threshold_Triplet / (Config.threshold_Triplet + Dissimilarity_Triplet)
        # imshow(torchvision.utils.make_grid(concatenated), 'similarity: {:.2f}%'.format(similarity_percentage * 100))

    return ground_truth, predict_Contra, predict_Triplet_All, predict_Triplet_Hard


def test_net(model):
    if model == "[LeNet]":
        net_Contra = LeNet().cuda()
        net_Triplet_All = LeNet().cuda()
        net_Triplet_Hard = LeNet().cuda()
    elif model == "[ResNet]":
        net_Contra = models.resnet18().cuda()
        net_Triplet_All = models.resnet18().cuda()
        net_Triplet_Hard = models.resnet18().cuda()

        fc_features = net_Contra.fc.in_features
        net_Contra.fc = nn.Linear(fc_features, 10).cuda()
        net_Triplet_All.fc = nn.Linear(fc_features, 10).cuda()
        net_Triplet_Hard.fc = nn.Linear(fc_features, 10).cuda()

    net_Contra.load_state_dict(torch.load(Config.params_root1))
    net_Triplet_All.load_state_dict(torch.load(Config.params_root2))
    net_Triplet_Hard.load_state_dict(torch.load(Config.params_root3))

    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = PairDataset(imageFolderDataset=folder_dataset_test,
                                  transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                transforms.ToTensor()]), should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)

    ground_truth, predict_Contra, predict_Triplet_All, predict_Triplet_Hard = evaluate(net_Contra, net_Triplet_All,
                                                                                       net_Triplet_Hard,
                                                                                       test_dataloader)
    acc_mat_contra = np.equal(ground_truth, predict_Contra)
    acc_contra = np.mean(acc_mat_contra)

    acc_mat_triplet_All = np.equal(ground_truth, predict_Triplet_All)
    acc_triplet_All = np.mean(acc_mat_triplet_All)

    acc_mat_triplet_Hard = np.equal(ground_truth, predict_Triplet_Hard)
    acc_triplet_Hard = np.mean(acc_mat_triplet_Hard)
    print("============================================")
    print("Test size is : %s" % Config.test_batch_size)
    print("The acc_contra is : %s" % acc_contra)
    print("The acc_triplet_Batch_All is : %s" % acc_triplet_All)
    print("The acc_triplet_Batch_Hard is : %s" % acc_triplet_Hard)
    print("============================================")


def evaluate_onecase(model, img0_path, img1_path):
    if model == "[LeNet]":
        net_Triplet_All = LeNet().cuda()
    elif model == "[ResNet]":
        net_Triplet_All = models.resnet18().cuda()

        fc_features = net_Triplet_All.fc.in_features
        net_Triplet_All.fc = nn.Linear(fc_features, 10).cuda()

    net_Triplet_All.load_state_dict(torch.load(Config.params_root2))

    img0 = Image.open(img0_path).resize((100, 100), Image.BILINEAR)
    img1 = Image.open(img1_path).resize((100, 100), Image.BILINEAR)
    npimg0 = np.array(img0, dtype=np.float32).reshape((1, 1, 100, 100))
    npimg1 = np.array(img1, dtype=np.float32).reshape((1, 1, 100, 100))

    # 通过net_Triplet_All计算输出
    embeddings0 = net_Triplet_All(torch.from_numpy(npimg0).cuda())
    embeddings1 = net_Triplet_All(torch.from_numpy(npimg1).cuda())
    euclidean_distance_Triplet = F.pairwise_distance(embeddings0, embeddings1)
    Dissimilarity_Triplet = euclidean_distance_Triplet.item()
    similarity_percentage = Config.threshold_Triplet / (Config.threshold_Triplet + Dissimilarity_Triplet)
    return similarity_percentage * 100


if __name__ == '__main__':
    test_net(model=Config.model)
