import cv2
img1 = cv2.imread("add1.png")
img2 = cv2.imread("gray.png",0)
'''
cv2.cvtColor（input_image,flag）
--参数flag决定了转换的类型，flag可以是：
cv2.COLOR_BGR2GRAY， cv2.COLOR_RGB2GRAY
cv2.COLOR_BGR2HSV，cv2.COLOR_GRAY2RGB
'''
img3 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
img4 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
cv2.imshow("rgb_to_gray",img3)
cv2.imshow("gray_to_rgb",img4)
cv2.waitKey(0)

img = cv2.imread("add1.png")
ret,one = cv2.cv2.threshold(img,175,255,cv2.THRESH_BINARY)
cv2.imshow("yuzhi1",one)
ret,two = cv2.cv2.threshold(img,175,255,cv2.THRESH_BINARY_INV)
cv2.imshow("yuzhi2",two)
ret,three = cv2.cv2.threshold(img,175,255,cv2.THRESH_TRUNC)
cv2.imshow("yuzhi3",three)
cv2.waitKey(0)
'''
ret, dst = cv2.threshold(src,threshold,maxval type)
img就是输入图像，threshold就是阈值，maxval表示阈值类型，
不同的阈值类型，将大于设置的threshold的像素转化方向不同（即有的可以将大于阈值的像素转化为0，有的则是转化为255）
cv2.THRESH_BINARY 大于阈值的部分被置为255，小于部分被置为0
cv2.THRESH_BINARY_INV 大于阈值部分被置为0，小于部分被置为255
cv2.THRESH_TRUNC 大于阈值部分被置为threshold，小于部分保持原样
cv2.THRESH_TOZERO 小于阈值部分被置为0，大于部分保持不变
cv2.THRESH_TOZERO_INV 大于阈值部分被置为0，小于部分保持不变
cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值，并作为第一个返回值ret返回。
'''