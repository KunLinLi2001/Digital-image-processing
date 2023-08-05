import cv2
from matplotlib import pyplot as plt
import numpy as np
def Global_threshold_processing(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 构建直方图
    T = round(img.mean()) # 计算图像灰度的中值作为初始阈值T
    num = img.shape[0]*img.shape[1] # 计算像素点的个数，为计算阈值两侧均值做准备
    while True:
        num_l,num_r,sum_l,sum_r = 0,0,0,0 # 直方图中阈值T两侧像素点的个数
        for i in range(0,T):
            num_l += hist[i,0] # 阈值左侧像素数量
            sum_l += hist[i,0]*i
        for i in range(T,256):
            num_r += hist[i,0] # 阈值左侧像素数量
            sum_r += hist[i,0]*i
        T1 = round(sum_l/num_l) # 左端平均灰度值
        T2 = round(sum_r/num_r) # 右端平均灰度值
        tnew = round((T1+T2)/2) # 计算出的新阈值
        if T==tnew: # 两侧的平均灰度值相同
            break
        else:
            T = tnew # 更新阈值
    ret,dst = cv2.threshold(img,T,255,cv2.THRESH_BINARY) # 阈值分割
    return dst

img = cv2.imread("C:\\tmp\\four\\4.2.jpg",0)
ret,dst = cv2.threshold(img,180,255,cv2.THRESH_BINARY) # 阈值分割
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # 5*5卷积核(结构元)
open = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel) # 开运算
close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel) # 闭运算

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(dst,cmap='gray')
plt.title('split'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(open,cmap='gray')
plt.title('open'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(close,cmap='gray')
plt.title('open->close'), plt.xticks([]), plt.yticks([])

plt.show()