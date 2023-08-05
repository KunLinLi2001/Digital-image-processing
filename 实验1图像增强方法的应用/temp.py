# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:09:33 2022

@author: joker
"""
import cv2
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import util
from PIL import Image
import random

def jiaoyan(img,prob):
    output = np.zeros(img.shape,np.uint8)#创建一个与原图像相同的白板
    thres = 1 - prob
    for i in range(img.shape[0]):
       for j in range(img.shape[1]):
                rdn = random.random()#随机生成0-1之间的数字
                if rdn < prob:#如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                    output[i][j] = 0
                elif rdn > thres:#如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]#其他情况像素点不变
    return output

def gaosi(img,var,mean):#var方差，mena均值
    img = np.array(img/255, dtype=float) #生成一个数列
    noise = np.random.normal(mean, var ** 0.5, img.shape) #生成一个均值为0.1，方差为0.01的正态分布
    out = img + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out

def meanlvbo(img):
    blur = cv2.blur(img, (3,3))
    return blur

def middlelvbo(img):
    medianblur = cv2.medianBlur(img,5)
    return medianblur

def ditonglvbo(img,D0,n):  
    img = np.float32(img)  
    length, width = img.shape  # 傅里叶变换  
    img_fft = np.fft.fft2(img)  # 零频率分量移到频谱中心  
    img_fftshift = np.fft.fftshift(img_fft)  # 所有像素点坐标的坐标矩阵为M和N  
    M, N = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-length // 2, length // 2))  # 每个坐标的频率计算出来  
    D = np.sqrt(M ** 2 + N ** 2)  # 巴特沃斯低通滤波器的转移函数  
    H = 1/(1+np.power(D/D0,2*n))  # 与原图像相乘  
    img_fftshift = H * img_fftshift  # 将频域中心移回原位置  
    img_ifftshift = np.fft.ifftshift(img_fftshift)  # 傅里叶逆变换  
    img_ifft = np.fft.ifft2(img_ifftshift)  # 保留实部  
    res = np.real(img_ifft)  # 转化成0-255的整型灰度值矩阵  
    res = np.uint8(np.clip(res, 0, 255))  
    return res  
# 巴特沃斯高通滤波器  
def gaotonglvbo(img,D0,n):   
    img = np.float32(img)  
    length, width = img.shape  # 傅里叶变换  
    img_fft = np.fft.fft2(img)  # 零频率分量移到频谱中心  
    img_fftshift = np.fft.fftshift(img_fft)  # 所有像素点坐标的坐标矩阵为M和N  
    M, N = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-length // 2, length // 2))  # 每个坐标的频率计算出来  
    D = np.sqrt(M ** 2 + N ** 2)  # 巴特沃斯低通滤波器的转移函数  
    H = 1 / (1 + np.power(D0/ D, 2 * n))  # 与原图像相乘  
    img_fftshift = H * img_fftshift  # 将频域中心移回原位置  
    img_ifftshift = np.fft.ifftshift(img_fftshift)  # 傅里叶逆变换  
    img_ifft = np.fft.ifft2(img_ifftshift)  # 保留实部  
    res = np.real(img_ifft)  # 转化成0-255的整型灰度值矩阵  
    res = np.uint8(np.clip(res, 0, 255))  
    return res  

img = cv2.imread("2.bmp",0)

noise = jiaoyan(img,0.01)
noise = gaosi(noise,0.01,0.1)
cv2.imshow("noise",noise)
cv2.waitKey(0)

noise = np.uint8(np.clip(noise, 0, 255))
#首先对图像依次进行均值滤波，中值滤波和低通滤波
noise_1 = cv2.medianBlur(noise,3)
cv2.imshow("aa",noise_1)
cv2.waitKey(0)

noise_2 = cv2.blur(noise_1, (4,4))
noise_3 = ditonglvbo(noise_2)

plt.subplot(1, 4, 1), plt.imshow(noise, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1, 4, 2), plt.imshow(noise_1, cmap='gray')  
plt.title('junzhilvbo'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1, 4, 3), plt.imshow(noise_2, cmap='gray')  
plt.title('zhongzhilvbo'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1, 4, 4), plt.imshow(noise_3, cmap='gray')  
plt.title('ditonglvbo'), plt.xticks([]), plt.yticks([])
