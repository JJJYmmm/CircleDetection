'''
Author : JoeHuang
Time : 2023/7/6
'''

import numpy as np
import math

class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y**2 + self.x**2))
        self.step = step
        self.vote_matrix = np.zeros([math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print ('Hough_transform_algorithm')
        # ------------- write your code bellow ----------------
        for i in range(1, self.y - 1):
            for j in range(1, self.x -1):
                if self.img[i][j] > 0: # img是canny处理过的边缘图
                    y = i
                    x = j
                    r = 0
                    # line 1 
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y/self.step)][math.floor(x/self.step)][math.floor(r / self.step)] += 1
                        y = y + self.step * self.angle[i][j]
                        x = x + self.step
                        r = r + math.sqrt((self.step * self.angle[i][j])**2 + self.step**2)
                    # line 2
                    y = i - self.step * self.angle[i][j]
                    x = j - self.step
                    r = math.sqrt((self.step*self.angle[i][j])**2+self.step**2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y/self.step)][math.floor(x/self.step)][math.floor(r / self.step)] += 1
                        y = y - self.step * self.angle[i][j]
                        x = x - self.step
                        r = r + math.sqrt((self.step * self.angle[i][j])**2 + self.step**2) 

        # ------------- write your code above ----------------
        return self.vote_matrix


    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print ('Select_Circle')
        # ------------- write your code bellow ----------------
        waitlist = []
        for i in range(0,math.ceil(self.y/self.step)):
            for j in range(0,math.ceil(self.x/self.step)):
                for r in range(0,math.ceil(self.radius/self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i*self.step + self.step / 2
                        x = j*self.step + self.step / 2
                        r = r*self.step + self.step / 2
                        waitlist.append((math.ceil(x),math.ceil(y),math.ceil(r)))
        if len(waitlist) == 0:
            print("No Circles in the picture,considering reduce the threshold.")
            return
        x, y, r = waitlist[0]
        possible = []
        middle = []
        for circle in waitlist:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1],circle[2]]) # 相似的放在possible 之所以可以这么做 是因为waitlist生成时最里循环是r 可以保证坐标位置相同的⚪在waitlist中连续
            else:
                result = np.array(possible).mean(axis = 0)
                middle.append((result[0],result[1],result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x,y,r])

        result = np.array(possible).mean(axis=0) #处理尾巴
        middle.append((result[0],result[1],result[2]))

        def takeFirst(elem):
            return elem[0]
        
        middle.sort(key=takeFirst)
        x, y, r = middle[0]
        possible = []
        for circle in middle:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)
                print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)
        print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
        self.circles.append((result[0], result[1], result[2]))

        # ------------- write your code above ----------------


    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles