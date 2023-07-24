import cv2
img= cv2.imread('pictures/ying.jpg')
cv2.imshow('img',img)
h=img.shape[0]
w=img.shape[1]
dsize=(h,w)
m = cv2.getRotationMatrix2D((w/2,h/2),60,0.5)
img2=cv2.warpAffine(img,m,dsize)
cv2.imshow('rotation',img2)
cv2.waitKey(0)
'''获取图像的旋转矩阵：
 cv2.getRotationMatrix2D(center,angle,scale)
--center表示原图像中作为旋转中心的坐标
 --angle表示旋转角度，正数为逆时针旋转，负数为顺时针旋转
 --scale表示目标图像与原图像的大小比例'''