import cv2 as cv
import numpy as np
bird = cv.imread(r'D:\img_test\opencv.jpg')
if bird is None:
    print("路径问题")
else:
    blur = cv.blur(bird,(5,5))
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    low_blue = np.array([100,0,0])
    high_blue = np.array([150,255,255])
#可实现二值化功能（类似threshold()函数）可以同时针对多通道进行操作
mask = cv.inRange(hsv,low_blue,high_blue)
res = cv.bitwise_and(bird,bird,mask=mask)
cv.imshow("img",res)
cv.waitKey(0)
