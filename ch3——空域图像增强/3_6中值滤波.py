import cv2
import skimage
import numpy as np
img = cv2.imread("aoligei.jpg")
cv2.imshow("11",img)
'''生成两种噪声'''
noise1 = skimage.util.random_noise(img,mode='gaussian',var=0.01)
noise2 = skimage.util.random_noise(img,mode='s&p',salt_vs_pepper=0.5)
noise1 = noise1*255
noise2 = noise2*255
noise1 = noise1.astype(np.uint8)
noise2 = noise1.astype(np.uint8)
cv2.imshow("gaosi",noise1)
cv2.imshow("jiaoyan",noise2)
'''中止滤波'''
res1 = cv2.medianBlur(noise1,5)
res2 = cv2.medianBlur(noise2,5)
cv2.imshow("res1",res1)
cv2.imshow("res2",res2)
cv2.waitKey(0)