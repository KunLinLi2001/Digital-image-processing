import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("wife.jpg")
# 为了计算转换成float32类型
img = img.astype('float32')
(B,G,R) = cv2.split(img)
# 计算HSI
I = (R+G+B)/3
Min = np.where(B>=G,G,B)
Min = np.where(Min<=R,Min,R)
S = 1-(3/(R+G+B))*Min
angle = np.arccos((2*R-G-B)/(2*np.sqrt(((R-G)**2)+(R-B)*(G-B))))
H = np.where(G>=B,angle,2*np.pi-angle)
H = np.clip(H,0,2*np.pi)
print(H)
imgHSI = np.copy(img)
imgHSI[:, :, 0], imgHSI[:, :, 1], imgHSI[:, :, 2] = H,S,I
plt.subplot(1,4,1),plt.imshow(H,cmap='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2),plt.imshow(S,cmap='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3),plt.imshow(I,cmap='gray')
plt.title('I'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4),plt.imshow(imgHSI)
plt.title('HSI'), plt.xticks([]), plt.yticks([])

# 更改一下HSI的数值
I = I
H = H
S = S
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        HH = H[i][j]
        SS = S[i][j]
        II = I[i][j]
        if (HH >= 0) & (HH < (np.pi * (2 / 3))):
            B[i][j] = II * (1 - SS) if II * (1 - SS) <=255  else 255
            R[i][j] = II * (1 + ((SS * np.cos(HH)) / np.cos(np.pi * (1 / 3) - HH))) if II * (1 + ((SS * np.cos(HH)) / np.cos(np.pi * (1 / 3) - HH)))<=255 else 255
            G[i][j] = 3 * II - (B[i][j] + R[i][j]) if 3 * II - (B[i][j] + R[i][j])<=255 else 255
        elif (HH >= (np.pi * (2 / 3))) & (HH < np.pi * (4 / 3)):
            R[i][j] = II * (1 - SS) if II * (1 - SS)<=255 else 255
            G[i][j] = II * (1 + ((SS * np.cos(HH - np.pi * (2 / 3))) / np.cos(np.pi * (1 / 2) - HH))) if II * (1 + ((SS * np.cos(HH - np.pi * (2 / 3))) / np.cos(np.pi * (1 / 2) - HH)))<=255 else 255
            B[i][j] = 3 * II - (G[i][j] + R[i][j]) if 3 * II - (G[i][j] + R[i][j])<=255 else 255
        elif (HH >= (np.pi * (2 / 3))) & (HH < (np.pi * 2)):
            G[i][j] = II * (1 - SS) if II * (1 - SS)<=255 else 255
            B[i][j] = II * (1 + ((SS * np.cos(HH - np.pi * (4 / 3))) / np.cos(np.pi * (10 / 9) - HH))) if II * (1 + ((SS * np.cos(HH - np.pi * (4 / 3))) / np.cos(np.pi * (10 / 9) - HH)))<=255 else 255
            R[i][j] = 3 * II - (G[i][j] + B[i][j]) if 3 * II - (G[i][j] + B[i][j])<=255 else 255
G = G.astype(np.uint8)
G = np.clip(G,0,255)
R = R.astype(np.uint8)
R = np.clip(R,0,255)
B = B.astype(np.uint8)
B = np.clip(B,0,255)
iimg = cv2.merge([B,G,R])
cv2.imshow("iimg",iimg)
plt.show()
cv2.waitKey(0)


