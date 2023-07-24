import cv2
import numpy as np
filename = "add1.png"
img = cv2.imread(filename)
print(img.shape)# (1276, 1702, 3) 返回值为一个元组，代表宽、长、通道数
print(img.shape[:2])# (1276, 1702) 彩色图片的宽、长
print(img.shape[:3])# (1276, 1702, 3) 彩色图片的宽、长、通道
print(img.shape[0])
print(img.shape[1])
print(img.shape[2])
print(img)
# 三个不同性质元素
print(img.size) # 6515256 像素数1276*1702*3
print(img.dtype) # uint8 图像类型
print("----------------------------------")
filename = "add1.png"
img = cv2.imread(filename,0)
print(img.shape)
print(img.shape[:2])
print(img.shape[0])
print(img.shape[1])
print(img)
# 三个不同性质元素
print(img.size) # 6515256 像素数1276*1702*3
print(img.dtype) # uint8 图像类型