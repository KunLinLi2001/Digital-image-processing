import cv2
img = cv2.imread("pictures/ying.jpg")
res = cv2.resize(img,None,fx=2, fy=0.5, interpolation = cv2.INTER_CUBIC)
# height, width = img.shape[:2]
# res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
'''
cv2.resize(src,dsize，fx,fy,interpolation)
--interpolation，插值方式；
cv2.INTER_NEAREST，cv.INTER_LINEAR，cv2.INTER_CUBIC等；
默认情况下，所有的放缩都使用cv.INTER_LINEAR
'''
cv2.imshow('dogegg',res)
cv2.waitKey(0)