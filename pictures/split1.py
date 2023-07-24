# 彩色图像分割：HSI用饱和度做模板，RGB设置不同阈值
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# HSI用饱和度做模板进行彩色图像分割 
img = cv.imread('C:\\tmp\\wife.jpg') #从指定文件加载图像并返回该图像的矩阵#
hsi = cv.cvtColor(img,cv.COLOR_BGR2HLS) #颜色空间转换#
plt.xlim([0,256])
low_r = np.array([156,43,46])
high_r = np.array([180,255,255])
mask = cv.inRange(hsi,low_r,high_r) #将阈值外的图像值变为0阈值内变为255#
res = cv.bitwise_and(img,img,mask=mask) #对二进制数据进行与操作#
cv.imshow("img",res)
cv.waitKey(0)