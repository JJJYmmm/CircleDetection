# Circle Detection

本项目使用 Canny + Hough 对图像中的圆形进行检测。

文件列表如下：

- main.py : run detection program
- my_canny.py : canny算法实现，得到图像的梯度图/梯度方向图
- my_hough.py : hough算法实现，实现通过参数空间的投票算法进行圆形的数学建模

使用方法：

- 检测图像放在picture_source文件夹下，命名为picture.jpg(或修改main.py中的Path路径)
- Canny/Hough检测结果放在picture_result文件夹下

测试结果：

左侧为canny算法结果，右侧为hough检测出的圆(原图上画出)

![image-20230706191741408](https://github.com/JJJYmmm/CircleDetection/blob/master/concat_result.jpg)

![image-20230706192044307](https://github.com/JJJYmmm/CircleDetection/blob/master/concat_result.jpg)
