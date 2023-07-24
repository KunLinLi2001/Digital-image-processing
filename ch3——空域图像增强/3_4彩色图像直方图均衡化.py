import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("wife.jpg")
cv2.imshow("src", img)
'''直方图均衡化'''
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
res = cv2.merge((bH, gH, rH))
cv2.imshow("res",res)
'''直方图绘制'''
color = ('b','g','r')
plt.figure(1)
for i,col in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
plt.xlim([0,256]) # 设置x轴数据显示范围
plt.figure(2)
for i,col in enumerate(color):
    hist = cv2.calcHist([res],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
plt.xlim([0,256]) # 设置x轴数据显示范围
plt.show()

cv2.waitKey(0)




# RGB2HSI
def RGB2HSI(img1):
    img1 = img1.astype('float32')
    b, g, r = img1[:, :, 0] / 255.0, img1[:, :, 1] / 255.0, img1[:, :, 2] / 255.0

    I = (r + g + b) / 3.0

    tem = np.where(b >= g, g, b)
    minValue = np.where(tem >= r, r, tem)
    S = 1 - (3 / (r + g + b)) * minValue

    num1 = 2 * r - g - b
    num2 = 2 * np.sqrt(((r - g) ** 2) + (r - b) * (g - b))
    deg = np.arccos(num1 / num2)
    H = np.where(g >= b, deg, 2 * np.pi - deg)

    resImg = np.zeros((img1.shape[0], img1.shape[1],
                       img1.shape[2]), dtype=np.float)
    resImg[:, :, 0], resImg[:, :, 1], resImg[:, :, 2] = H * 255, S * 255, I * 255
    resImg = resImg.astype('uint8')
    return resImg


def HSI2RGB(img):
    H1, S1, I1 = img[:, :, 0] / 255.0, img[:, :, 1] / 255.0, img[:, :, 2] / 255.0
    B = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')
    G = np.zeros((S1.shape[0], S1.shape[1]), dtype='float32')
    R = np.zeros((I1.shape[0], I1.shape[1]), dtype='float32')
    H = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')

    for i in range(H1.shape[0]):
        for j in range(H1.shape[1]):
            H = H1[i][j]
            S = S1[i][j]
            I = I1[i][j]
            if (H >= 0) & (H < (np.pi * (2 / 3))):
                B[i][j] = I * (1 - S)
                R[i][j] = I * (1 + ((S * np.cos(H)) / np.cos(np.pi * (1 / 3) - H)))
                G[i][j] = 3 * I - (B[i][j] + R[i][j])

            elif (H >= (np.pi * (2 / 3))) & (H < np.pi * (4 / 3)):
                R[i][j] = I * (1 - S)
                G[i][j] = I * (1 + ((S * np.cos(H - np.pi * (2 / 3))) / np.cos(np.pi * (1 / 2) - H)))
                B[i][j] = 3 * I - (G[i][j] + R[i][j])
            elif (H >= (np.pi * (2 / 3))) & (H < (np.pi * 2)):
                G[i][j] = I * (1 - S)
                B[i][j] = I * (1 + ((S * np.cos(H - np.pi * (4 / 3))) / np.cos(np.pi * (10 / 9) - H)))
                R[i][j] = 3 * I - (G[i][j] + B[i][j])
    img = cv2.merge((B * 255, G * 255, R * 255))
    img = img.astype('uint8')
    return img


img1 = cv2.imread('./t1.jpg')
show('img', img1)
res = RGB2HSI(img1)
show('RGB2HSI', res)
a = HSI2RGB(res)
show('HSI2RGB', a)
