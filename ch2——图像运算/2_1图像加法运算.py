import cv2
img1 = cv2.imread("add1.png")
img2 = cv2.imread("add2.jpg")
img3 = img1+img2 # 不是加法的意思
img4 = cv2.add(img1,img2) # 普通加法
img5 = cv2.addWeighted(img1,0.7,img2,0.3,0) # 加权加法
cv2.imshow("cui",img1)
cv2.imshow("erha",img2)
cv2.imshow("cui+erha",img3)
cv2.imshow("add",img4)
cv2.imshow("add_weight",img5)
cv2.waitKey(0)