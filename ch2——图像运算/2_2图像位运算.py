import cv2
img1 = cv2.imread("add1.png")
img2 = cv2.imread("add2.jpg")
img3 = cv2.bitwise_and(img1,img2) # 与
img4 = cv2.bitwise_or(img1,img2) # 或
img5 = cv2.bitwise_not(img1) # 非
img6 = cv2.bitwise_xor(img1,img2) # 异或
cv2.imshow("cui",img1)
cv2.imshow("erha",img2)
cv2.imshow("and",img3)
cv2.imshow("or",img4)
cv2.imshow("not",img5)
cv2.imshow("xor",img6)
cv2.waitKey(0)