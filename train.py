import math

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
from triplet_loss import *
import PIL.ImageOps
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F
import os
import torchvision.models as models


class Config():
    # 基于两套数据集训练下的模型参数
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
    # 若损失函数为TripletLoss，则定义数据采集方式{batch_all, batch_hard}，默认为batch_hard
    batch_strategy = batch_all_triplet_loss
    # 定义训练集路径
    training_dir = os.path.join("datasets", data_sets, "train")
    # 定义模型参数保存路径及名称
    params_dir = os.path.join("params", data_sets, model + __params_contra)

    # 定义超参数: 学习率大小、epoch大小、训练集batchsize大小、损失函数margin等
    lr = 1e-4
    train_number_epochs = 50
    train_batch_size = 64
    contrastive_loss_margin = 3.2
    triplet_loss_margin = 20.

    test_batch_size = 100
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


def show_pair(siamese_dataset):
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.xlabel('epoches(per {} epoch)'.format(1))
    plt.ylabel('loss')
    plt.title('Learning rate = ' + str(Config.lr))
    plt.show()


class PairDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # 如果为contrastive Loss, 则改为img1_tuple[1] != img0_tuple[1]
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SingleDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.data_set = imageFolderDataset.imgs
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        path, label = self.data_set[index]
        img = Image.open(path)
        # img = img.convert("L")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

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
        embeddings = self.cnn1(x)
        embeddings = embeddings.view(embeddings.size(0), -1)
        embeddings = self.fc1(embeddings)
        return embeddings


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, label):
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2)
        contrastive_loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return contrastive_loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=10.0, batch_strategy=batch_hard_triplet_loss, squared=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        self.batch_strategy = batch_strategy

    def forward(self, embeddings, labels):
        triplet_loss = self.batch_strategy(labels=labels, embeddings=embeddings, margin=self.margin,
                                           squared=self.squared)
        return triplet_loss


def train_contrastive(model):
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = PairDataset(imageFolderDataset=folder_dataset,
                                  transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                transforms.ToTensor()]), should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=Config.train_batch_size)
    pair_dataset = PairDataset(imageFolderDataset=folder_dataset,
                               transform=transforms.Compose([transforms.Resize((100, 100)),
                                                             transforms.ToTensor()]), should_invert=False)
    eva_dataloader = DataLoader(pair_dataset, shuffle=True, num_workers=0, batch_size=1)
    if model == "[LeNet]":
        net = LeNet().cuda()
    elif model == "[ResNet]":
        net = models.resnet18().cuda()
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, 5).cuda()

    criterion = ContrastiveLoss(margin=Config.contrastive_loss_margin)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr, betas=(0.9, 0.99))

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, Config.train_number_epochs):
        loss = 0
        for i, data in enumerate(train_dataloader, 0):
            img1, img2, labels = data
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.cuda()
            # img.shape=[m,1,100,100]      labels.shape=[m,1]
            optimizer.zero_grad()
            embeddings1 = net(img1)
            embeddings2 = net(img2)
            loss = criterion(embeddings1, embeddings2, labels)
            loss.backward()
            optimizer.step()

        ground, predict = compute_accuracy(net, eva_dataloader, Config.threshold_Contra)
        acc_mat_contra = np.equal(ground, predict)
        acc_contra = np.mean(acc_mat_contra)
        print("Epoch number {}\n Current loss {}".format(epoch, loss.item()))
        print(" The acc_triplet is : %s" % acc_contra)
        print("============================================")
        counter.append(epoch)
        loss_history.append(loss.item())
    show_plot(counter, loss_history)
    torch.save(net.state_dict(), Config.params_dir)


def train_triplet(model):
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SingleDataset(imageFolderDataset=folder_dataset,
                                    transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                   transforms.ToTensor()]), should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=Config.train_batch_size)
    pair_dataset = PairDataset(imageFolderDataset=folder_dataset,
                               transform=transforms.Compose([transforms.Resize((100, 100)),
                                                             transforms.ToTensor()]), should_invert=False)
    eva_dataloader = DataLoader(pair_dataset, shuffle=True, num_workers=0, batch_size=1)
    if model == "[LeNet]":
        net = LeNet().cuda()
    elif model == "[ResNet]":
        net = models.resnet18().cuda()
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, 5).cuda()

    criterion = TripletLoss(margin=Config.triplet_loss_margin, batch_strategy=Config.batch_strategy)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr, betas=(0.9, 0.99))

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, Config.train_number_epochs):
        loss = 0
        for i, data in enumerate(train_dataloader, 0):
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            # imgs.shape=[m,1,100,100]      labels.shape=[m,1]
            optimizer.zero_grad()
            embeddings = net(imgs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
        ground, predict = compute_accuracy(net, eva_dataloader, Config.threshold_Triplet)
        acc_mat_triplet = np.equal(ground, predict)
        acc_triplet = np.mean(acc_mat_triplet)
        print("Epoch number {}\n Current loss {}".format(epoch, loss.item()))
        print(" The acc_triplet is : %s" % acc_triplet)
        print("============================================")
        counter.append(epoch)
        loss_history.append(loss.item())
    show_plot(counter, loss_history)
    torch.save(net.state_dict(), Config.params_dir)


def compute_accuracy(model, test_dataloader, threshold):
    ground = []
    predict = list()

    for i in range(Config.test_batch_size):
        # 随机产生一组图片对，若img0和img1为相同类别，则label=1
        data = next(iter(test_dataloader))
        img0, img1, label = data
        img0 = img0.cuda()
        img1 = img1.cuda()
        label = label.cuda()

        ground.append(torch.squeeze(label.int()).bool())

        # 通过model计算输出
        embeddings0 = model(img0)
        embeddings1 = model(img1)
        euclidean_distance = F.pairwise_distance(embeddings0, embeddings1)
        dissimilarity = euclidean_distance.item()
        predict.append(dissimilarity <= threshold)
    return ground, predict


if __name__ == '__main__':
    train_contrastive(model=Config.model)
