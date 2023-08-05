'''对给定的 leaf 图片进行顶帽变换和底帽变换，比较结果。'''
import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread("C:\\tmp\\four\\4.5.jpg",0)
plt.subplot(3,2,1),plt.imshow(img,cmap='gray')
plt.title('Normal'), plt.xticks([]), plt.yticks([])

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # 5*5卷积核(结构元)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)) # 25*25卷积核(结构元)
tophat1 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel1) # 顶帽运算加强暗背景下的亮细节
tophat2 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel2)

blackhat1 = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel1) # 底帽运算加强亮背景处的暗细节
blackhat2 = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel2)

plt.subplot(3,2,3),plt.imshow(tophat1,cmap='gray')
plt.title('25*25 tophat'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(tophat2,cmap='gray')
plt.title('70*70 tophat'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(blackhat1,cmap='gray')
plt.title('25*25 blackhat'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(blackhat2,cmap='gray')
plt.title('70*70 blackhat'), plt.xticks([]), plt.yticks([])

plt.show()