import cv2
import skimage
img = cv2.imread("wife.jpg")
cv2.imshow("11",img)
'''生成两种噪声'''
noise1 = skimage.util.random_noise(img,mode='gaussian',var=0.01)
noise2 = skimage.util.random_noise(img,mode='s&p',salt_vs_pepper=0.5)
cv2.imshow("gaosi",noise1)
cv2.imshow("jiaoyan",noise2)
'''均值滤波'''
res1 = cv2.blur(noise1,(3,3))
res2 = cv2.blur(noise2,(3,3))
'''高斯滤波'''
res3 = cv2.GaussianBlur(noise1,(3,3),0,0)
res4 = cv2.GaussianBlur(noise2,(3,3),0,0)
'''输出'''
cv2.imshow("res1",res1)
cv2.imshow("res2",res2)
cv2.imshow("res3",res3)
cv2.imshow("res4",res4)
cv2.waitKey(0)
