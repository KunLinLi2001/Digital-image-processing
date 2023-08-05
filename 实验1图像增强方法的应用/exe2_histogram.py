'''
2.分别显示原图像和添加噪声后图像的直方图，
去除图像噪声后对去噪后的图像进行直方图均衡化，显示均衡化后的图像和直方图。
'''
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
# numpy生成高斯噪声
def gaussian(img,var=0.01):
  img = np.array(img/255, dtype=float)
  noise = np.random.normal(0,var ** 0.5,img.shape)
  out = img + noise
  out = np.clip(out, 0, 1.0)
  out = np.uint8(out*255)
  return out
# numpy生成椒盐噪声
def pepper_salt(img,prob=0.1):
    wid,len = img.shape[0],img.shape[1]
    for i in range(0,wid):
        for j in range(0,len):
            rdm = random.random()
            if rdm<prob:
                img[i][j]=0
            elif rdm>(1-prob):
                img[i][j]=255
    return img
# 原正常灰度图
img = cv2.imread("2.bmp",0)
# 添加了高斯噪声的图像
noise_img = gaussian(img, 0.01)
# figure1显示原图像和添加噪声后图像的直方图
plt.figure(1)
plt.subplot(1,2,1),plt.hist(img.ravel(),256,[0,256])
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.hist(noise_img.ravel(),256,[0,256])
plt.title('add noise'), plt.xticks([]), plt.yticks([])
# 添加均值滤波去噪
noise_img = cv2.blur(noise_img,(5,5))
# 对两个图像分别进行直方图均衡化
res1 = cv2.equalizeHist(img)
res2 = cv2.equalizeHist(noise_img)
# figure2显示均衡化后的图像和直方图
plt.figure(2)
plt.subplot(2,2,1),plt.imshow(res1,cmap='gray')
plt.title('origin equalizeHist'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(res2,cmap='gray')
plt.title('clear noise equalizeHist'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.hist(res1.ravel(),256,[0,256])
plt.title('origin equalizeHist'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.hist(res2.ravel(),256,[0,256])
plt.title('clear noise equalizeHist'), plt.xticks([]), plt.yticks([])
plt.show()
