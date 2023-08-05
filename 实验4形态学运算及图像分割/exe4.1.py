import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:\\tmp\\four\\4.11.jpg",0)
'''将图像转换为二值图像'''
ret1,th1 = cv2.threshold(img,130,255,cv2.THRESH_BINARY)# 小于130置0,大于130置255
'''进行方形模板 3*3 和 5*5 的膨胀和腐蚀操作'''
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) # 3*3卷积核(结构元)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # 5*5卷积核(结构元)
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) # 7*7卷积核(结构元)
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) # 9*9卷积核(结构元)
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)) # 25*25卷积核(结构元)
expand1 = cv2.dilate(th1,kernel=kernel1) # 3*3膨胀操作 
expand2 = cv2.dilate(th1,kernel=kernel2) # 5*5膨胀操作 
expand3 = cv2.dilate(th1,kernel=kernel3) # 7*7膨胀操作 
expand4 = cv2.dilate(th1,kernel=kernel4) # 9*9膨胀操作 
expand5 = cv2.dilate(th1,kernel=kernel5) # 25*25膨胀操作
corrode1 = cv2.erode(th1,kernel=kernel1) # 3*3腐蚀操作
corrode2 = cv2.erode(th1,kernel=kernel2) # 5*5腐蚀操作
corrode3 = cv2.erode(th1,kernel=kernel3) # 7*7腐蚀操作
corrode4 = cv2.erode(th1,kernel=kernel4) # 9*9腐蚀操作
corrode5 = cv2.erode(th1,kernel=kernel5) # 25*25腐蚀操作

plt.subplot(2,6,1),plt.imshow(img,cmap='gray')
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,7),plt.imshow(th1,cmap='gray')
plt.title('Binary'), plt.xticks([]), plt.yticks([])
'''显示膨胀图像'''
plt.subplot(2,6,2),plt.imshow(expand1,cmap='gray')
plt.title('3*3 expand'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,3),plt.imshow(expand2,cmap='gray')
plt.title('5*5 expand'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,4),plt.imshow(expand3,cmap='gray')
plt.title('7*7 expand'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,5),plt.imshow(expand4,cmap='gray')
plt.title('9*9 expand'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,6),plt.imshow(expand5,cmap='gray')
plt.title('25*25 expand'), plt.xticks([]), plt.yticks([])
'''显示腐蚀图像'''
plt.subplot(2,6,8),plt.imshow(corrode1,cmap='gray')
plt.title('3*3 corrode'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,9),plt.imshow(corrode2,cmap='gray')
plt.title('5*5 corrode'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,10),plt.imshow(corrode3,cmap='gray')
plt.title('7*7 corrode'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,11),plt.imshow(corrode4,cmap='gray')
plt.title('9*9 corrode'), plt.xticks([]), plt.yticks([])
plt.subplot(2,6,12),plt.imshow(corrode5,cmap='gray')
plt.title('25*25 corrode'), plt.xticks([]), plt.yticks([])
# plt.figure(2)
# plt.subplot(1,2,1),plt.imshow(corrode2,cmap='gray')
# plt.title('7*7 corrode'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(corrode2,cmap='gray')
# plt.title('9*9 corrode'), plt.xticks([]), plt.yticks([])

plt.show()