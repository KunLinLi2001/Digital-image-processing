#图像平移
import cv2
import numpy as np
img= cv2.imread('pictures/ying.jpg')
cv2.imshow('img',img)
h=img.shape[0]
w=img.shape[1]
dsize=(h,w)
m=np.float32([[1,0,100],[0,1,50]])
img2=cv2.warpAffine(img,m,dsize)
cv2.imshow('py',img2)
cv2.waitKey(0)
'''
cv2.warpAffine(src,M,dsize[,dst[,flags[,borderMode[,borderValue]]]])
--src表示原始图像
 --M是一个2*3的变换矩阵，可以实现平移、旋转等多种操作
 --dsize为变换后的图像大小
 --flags为插值方式，默认为cv2.INTER_LINEAR
 --borderMode为边界类型，默认为cv2.BORDER_CONSTANT
 --borderValue为边界值，默认为0
'''