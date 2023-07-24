import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("wife.jpg")
(b,g,r) = cv2.split(img)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

(R,G,B) = cv2.split(imgRGB)
plt.figure(1)
plt.subplot(2,3,1),plt.imshow(imgRGB)
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(img)
plt.title('Error'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(R,cmap ='gray')
plt.title('NormalR'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(G,cmap ='gray')
plt.title('NormalG'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(B,cmap ='gray')
plt.title('NormalB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(R)
plt.title('ErrorR'), plt.xticks([]), plt.yticks([])
# 幂次运算增加亮度
plt.figure(2)
c,k = 20,0.46
brighter_r = c*np.power(R,k)
brighter_r = brighter_r.astype(np.uint8)
brighter_r = np.clip(brighter_r,0,255)

brighter_g = c*np.power(G,k)
brighter_g = brighter_g.astype(np.uint8)
brighter_g = np.clip(brighter_g,0,255)
print(brighter_g)

brighter_b = c*np.power(B,k)
brighter_b = brighter_b.astype(np.uint8)
brighter_b = np.clip(brighter_b,0,255)
print(brighter_b)

brighter_img = cv2.merge([brighter_r,brighter_g,brighter_b])
plt.subplot(2,2,1),plt.imshow(brighter_img)
plt.title('brighter_img'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(brighter_r,cmap = 'gray')
plt.title('brighter_r'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(brighter_r,cmap = 'gray')
plt.title('brighter_g'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(brighter_r,cmap = 'gray')
plt.title('brighter_b'), plt.xticks([]), plt.yticks([])

plt.show()






