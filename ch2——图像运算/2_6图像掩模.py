import cv2
img = cv2.imread('pictures/dundun.jpg',0)
mask = cv2.imread('pictures/mask.jpg',0)
res = cv2.multiply(img,mask)
print(mask)
img3 = cv2.bitwise_and(img,mask) # 与
img4 = cv2.bitwise_or(img,mask) # 或
img6 = cv2.bitwise_xor(img,mask) # 异或

cv2.imshow("dundun",img)
cv2.imshow("momo",mask)
cv2.imshow("get_it",res)
# cv2.imshow('yu',img3)
# cv2.imshow('huo',img4)
# cv2.imshow('yihuo',img6)
cv2.waitKey(0)