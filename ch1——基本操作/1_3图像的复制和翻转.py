import cv2
img = cv2.imread("pictures/ying.jpg")
# 图片的复制
imcopy = img.copy()
# 图片的反转
imcopy = cv2.flip(imcopy,-1)
'''
cv2.flip(img,flipcode)
--flipcode控制翻转效果，
flipcode = 0：沿x轴翻转；
flipcode=1：沿y轴翻转；
flipcode =-1：x,y轴同时
'''
cv2.imshow('dogegg',imcopy)
cv2.waitKey(0)
cv2.imwrite("dogegg1.jpg",imcopy,[int(cv2.IMWRITE_JPEG_QUALITY),95])