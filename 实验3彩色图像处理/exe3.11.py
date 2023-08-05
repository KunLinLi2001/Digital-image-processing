# 1.给定一幅彩色图像，分别显示RGB空间的各层，将RGB图像转换到HSI空间，显示HSI空间的各层。
import cv2
import numpy as np
from matplotlib import pyplot as plt

# RGB转HSI函数
def RGB2HSI(img):
    esp = 1e-8
    img = img.astype('float32')
    (R, G, B) = cv2.split(img)
    wid, len = img.shape[0], img.shape[1]
    R, G, B = R / 255, G / 255, B / 255
    '''计算I'''
    I = (R + G + B) / 3
    '''计算S'''
    # 解决计算S时分母出现0的问题
    R_G_B = R + G + B + esp  # 三层相加
    # 找到Min,计算对应像素点的S
    Min = np.where(B >= G, G, B)
    Min = np.where(Min <= R, Min, R)
    S = 1 - (3 / (R_G_B)) * Min
    '''计算H'''
    # 求解夹角：分母添加1e-8是避免分母为0的情况
    angle = np.arccos((0.5 * (R + R - G - B)) / (np.sqrt((R - G) * (R - G) + (R - B) * (G - B)) + esp))
    H = np.where(B > G, 2 * np.pi - angle, angle)
    H = H / (2 * np.pi)
    H = np.where(S == 0, 0, H)
    H = np.uint8(H * 255)
    S = np.uint8(S * 255)
    I = np.uint8(I * 255)
    return [H, S, I]

# 读入彩色图像(BGR排布)
img = cv2.imread("C:\\tmp\\6.14.jpg") # 两朵花的图像
(b, g, r) = cv2.split(img)
# 转化成RGB排布的彩色图像
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
(R, G, B) = cv2.split(imgRGB)

'''分别显示RGB空间的各层'''
plt.figure(1)
plt.subplot(2, 3, 1), plt.imshow(imgRGB)
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(img)
plt.title('Error'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(R, cmap='gray')
plt.title('NormalR'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(G, cmap='gray')
plt.title('NormalG'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(B, cmap='gray')
plt.title('NormalB'), plt.xticks([]), plt.yticks([])
# ErrorR是将单图层图像作为三图层不加cmap约束图像
plt.subplot(2, 3, 6), plt.imshow(R)
plt.title('ErrorR'), plt.xticks([]), plt.yticks([])

'''将RGB图像转换到HSI空间'''
[H, S, I] = RGB2HSI(imgRGB)
imgHSI = imgRGB.copy()
height, width = imgRGB.shape[0], imgRGB.shape[1]
for i in range(height):
    for j in range(width):
        imgHSI[i, j, 0] = H[i, j]
        imgHSI[i, j, 1] = S[i, j]
        imgHSI[i, j, 2] = I[i, j]

'''显示HSI空间的各层'''
plt.figure(2)
plt.subplot(2, 2, 1), plt.imshow(imgHSI)
plt.title('HSI'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(H, cmap='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(S, cmap='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(I, cmap='gray')
plt.title('I'), plt.xticks([]), plt.yticks([])
'''调用库函数转换HSV模型，检验自身所写的转换HSI代码效果方向正确'''
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
[h,s,v] = cv2.split(imgHSV)
'''显示HSV空间的各层'''
plt.figure(3)
plt.subplot(2,2,1),plt.imshow(imgHSV)
plt.title('HSV'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(h,cmap ='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(s,cmap ='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(v,cmap ='gray')
plt.title('V'), plt.xticks([]), plt.yticks([])
plt.show()