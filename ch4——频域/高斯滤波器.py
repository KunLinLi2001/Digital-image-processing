# 高斯低通滤波平滑
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("wife.jpg", 0)
rows, cols = img.shape  #获取图像大小
crow, ccol = int(rows/2), int(cols/2)   #取图像中心
img = np.log(img)   #1.图像对数运算
f = np.fft.fft2(img)    #2.2d傅里叶变换
fshift = np.fft.fftshift(f) #平移零频率分量至中心

for i in range(0, rows, 1): #3.乘二阶指数高通滤波的同态滤波函数
    for j in range(0, cols, 1):
        fshift[i, j] *= 0.4*(1-np.exp(-(int((np.sqrt((i-crow) ** 2+(j-ccol) ** 2)/30))**2)))+0.8

ishift =np.fft.ifftshift(fshift)    #将中心平移至边缘
iimg=np.fft.ifft2(ishift)   #4.傅里叶反变换
iimg=np.abs(iimg)   #取绝对值转化为空间域图像
iimg=np.exp(iimg)   #5.指数运算

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.subplot(122)
plt.imshow(iimg, cmap='gray')
plt.title('result')
plt.axis('off')
plt.show()