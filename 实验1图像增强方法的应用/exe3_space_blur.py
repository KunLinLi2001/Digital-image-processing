'''3、对于上述添加了噪声的图像综合运用空间域增强方法去除图像噪声，
    之后在平滑了的图像基础上强化图像边缘和细节。'''
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
# 对两幅图像分别添加高斯噪声和椒盐噪声
img = cv2.imread("2.bmp",0)
noise1 = gaussian(img,0.01)
noise2 = pepper_salt(img,0.2)
# 对高斯噪声添加均值滤波，椒盐噪声添加中值滤波
res1 = cv2.blur(noise1,(3,3))
res2 = cv2.medianBlur(noise2,5)
# 对平滑的图像进行锐化
res3 = cv2.Sobel(res1,cv2.CV_64F,1,0,ksize=5)
res4 = cv2.Sobel(res2,cv2.CV_64F,0,1,ksize=5)
'''没有选择拉普拉斯算子因为轮廓太浅了'''
# 输出平滑和锐化后的图像
plt.subplot(2, 2, 1), plt.imshow(res1, cmap='gray')
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(res2, cmap='gray')
plt.title('medianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(res3, cmap='gray')
plt.title('blur-Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(res4, cmap='gray')
plt.title('medianBlur-Sobel'), plt.xticks([]), plt.yticks([])
plt.show()










