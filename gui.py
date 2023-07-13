# -*- coding: utf-8 -*-

import tkinter
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk

from test import evaluate_onecase


class ImageLabelingTool:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Compare Faces')
        self.root.resizable(False, False)  # 固定窗口大小

        self.windowWidth = 1280
        self.windowHeight = 720
        self.screenWidth, self.screenHeight = self.root.maxsize()  # 获得屏幕宽和高
        geometry_param = '%dx%d+%d+%d' % (
            self.windowWidth, self.windowHeight, (self.screenWidth - self.windowWidth) / 2,
            (self.screenHeight - self.windowHeight) / 2)  # 设置窗口大小及偏移坐标
        self.root.geometry(geometry_param)

        # 创建主菜单
        self.main_menu = tkinter.Menu(self.root)
        self.main_menu.add_command(label="Load Faces", command=self.open_images)
        self.main_menu.add_command(label="Compute Similarity", command=self.compare_images)
        self.root.config(menu=self.main_menu)

        # 创建组件
        self.main_window = tkinter.PanedWindow(self.root, orient='horizontal', sashwidth=10)

        self.canvas_width = 640
        self.canvas_height = 600
        self.left_frame = tkinter.LabelFrame(self.main_window, text='Face 1')
        self.left_frame.pack()

        self.left_canvas = tkinter.Canvas(self.left_frame, width=self.canvas_width, height=self.canvas_height)
        self.left_canvas.pack()

        self.right_frame = tkinter.LabelFrame(self.main_window, text='Face 2')
        self.right_frame.pack()

        self.right_canvas = tkinter.Canvas(self.right_frame, width=self.canvas_width, height=self.canvas_height)
        self.right_canvas.pack()

        self.main_window.add(self.left_frame)
        self.main_window.add(self.right_frame)

        # 填满整个界面
        self.main_window.pack(padx=20, pady=5, fill='both', expand='yes')

    def open_images(self):
        # 选择左侧图片
        self.left_filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
        if self.left_filename:
            self.load_left_image(self.left_filename)

        # 选择右侧图片
        self.right_filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
        if self.right_filename:
            self.load_right_image(self.right_filename)

    def load_left_image(self, filename):
        self.left_image = Image.open(filename)
        self.left_image = self.left_image.resize((self.canvas_width, self.canvas_height))
        self.left_photo = ImageTk.PhotoImage(self.left_image)
        self.left_canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=self.left_photo,
                                      anchor='c')

    def load_right_image(self, filename):
        self.right_image = Image.open(filename)
        self.right_image = self.right_image.resize((self.canvas_width, self.canvas_height))
        self.right_photo = ImageTk.PhotoImage(self.right_image)
        self.right_canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=self.right_photo,
                                       anchor='c')

    def compare_images(self):
        # 计算相似度并更新相似度文本标签
        similarity_score = evaluate_onecase('[LeNet]', self.left_filename, self.right_filename)
        messagebox.showinfo("Result", "The similarity of this pair of faces is {:.2f}%".format(similarity_score))

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ImageLabelingTool()
    app.run()
