import cv2
import numpy as np
img= cv2.imread('dogegg.jpg')
cv2.imshow('img',img)
h=img.shape[0]
w=img.shape[1]
dsize=(h,w)
src=np.float32([[0,0],[w-10,0],[0,h-10]]) #取原图像3个顶点
dst=np.float32([[50,50],[w-100,80],[100,h-100]])#对应的目标点坐标
m = cv2.getAffineTransform(src,dst)
img2=cv2.warpAffine(img,m,dsize)#执行变换
cv2.imshow('3point',img2)
cv2.waitKey(0)
'''
三点映射变换：将图像转换为任意的平行四边形
 cv2.getAffineTransform(src,dst)
--src为原图像中3个点的坐标
 --dst为原图像中3个点在目标图像中的对应坐标
该函数将src和dst中的3个点作为平行四边形左上角、右上角
和左下角的三个点，按照这三个点之间的坐标变换关系计算
所有像素的转换矩阵。
'''