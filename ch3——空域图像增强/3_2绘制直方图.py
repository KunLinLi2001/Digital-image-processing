import cv2
from matplotlib import pyplot as plt
import numpy as np
img1 = cv2.imread("wife.jpg")
'''获取直方图：cv.calcHist()'''
hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([img1],[1],None,[256],[0,256])
hist3 = cv2.calcHist([img1],[2],None,[256],[0,256])
print(hist1)
'''绘制灰度直方图'''
img2 = cv2.imread("aoligei.jpg",0)
cv2.imshow("aaa",img2)
plt.hist(img2.ravel(),256,[0,256])
plt.show()
'''绘制彩色直方图'''
color = ('b','g','r')
for i,col in enumerate(color):
    hist = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
plt.xlim([0,256]) # 设置x轴数据显示范围
plt.show()

cv2.waitKey(0)