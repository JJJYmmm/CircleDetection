"""
Author : JoeHuang
Date : 2023/7/6
"""
import cv2
import numpy as np


class Canny:
    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        """
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        """
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        """
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        """
        print("Get_gradient_img")
        # ------------- write your code bellow ----------------
        grad_x_img = np.zeros([self.y, self.x], dtype=np.float)
        grad_y_img = np.zeros([self.y, self.x], dtype=np.float)
        for i in range(0, self.x):
            for j in range(0, self.y):
                if j == 0:
                    grad_y_img[j][i] = 1
                else:
                    grad_y_img[j][i] = np.sum(
                        np.array([[self.img[j - 1][i]], [self.img[j][i]]])
                        * self.y_kernal
                    )
                if i == 0:
                    grad_x_img[j][i] = 1
                else:
                    grad_x_img[j][i] = np.sum(
                        np.array([self.img[j][i - 1], self.img[j][i]]) * self.x_kernal
                    )

        grad_img, self.angle = cv2.cartToPolar(grad_x_img, grad_y_img)
        self.angle = np.tan(self.angle)
        self.img = grad_img.astype(np.uint8)
        # ------------- write your code above ----------------
        return self.img

    def Non_maximum_suppression(self):
        """
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        """
        print("Non_maximum_suppression")
        # ------------- write your code bellow ----------------
        result = np.zeros([self.y, self.x])
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if abs(self.img[i][j]) <= 4:  # 跳过非边缘点
                    result[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:
                    weight = 1 / abs(self.angle[i][j])  # 注意这里权重要取反 weight = |dx|/|dy|
                    # https://ai-chen.github.io/%E4%BC%A0%E7%BB%9F%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95/2019/08/21/Canny-%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.html
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    # g1 g2
                    #    C
                    #    g4 g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #    g2 g1
                    #    C
                    # g3 g4
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                else:
                    weight = abs(self.angle[i][j])
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    # g1
                    # g2 C g4
                    #      g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #      g3
                    # g2 C g4
                    # g1
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]

                temp1 = weight * gradient1 + (1 - weight) * gradient2
                temp2 = weight * gradient3 + (1 - weight) * gradient4
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0

        self.img = result
        # ------------- write your code above ----------------
        return self.img

    def Hysteresis_thresholding(self):
        """
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        """
        print("Hysteresis_thresholding")
        # ------------- write your code bellow ----------------
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:  # 这里条件与nms相反，因为延申方向是梯度的垂直方向
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        # g1 g2
                        #    C
                        #    g4 g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #    g2 g1
                        #    C
                        # g3 g4
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        # g1
                        # g2 C g4
                        #      g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #      g3
                        # g2 C g4
                        # g1
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold

        # ------------- write your code above ----------------
        return self.img

    def canny_algorithm(self):
        """
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        """
        self.img = cv2.GaussianBlur(
            self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0
        )
        self.Get_gradient_img()
        self.img_origin = self.img.copy()  # 此时复制的是梯度图
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img