import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("aoligei.jpg",0)
cv2.imshow("src",img)
plt.figure(1)
plt.hist(img.ravel(),256,[0,256])
dst = cv2.equalizeHist(img)
cv2.imshow("dst",dst)
plt.figure(2)
plt.hist(dst.ravel(),256,[0,256])
plt.show()
cv2.waitKey(0)


