from skimage import data, color, io
from matplotlib import pyplot as plt
import numpy as np
import math
import sys


# RGB to HSI
def rgb2hsi(r, g, b):
    r = r / 255
    g = g / 255
    b = b / 255
    num = 0.5 * ((r - g) + (r - b))
    den = ((r - g) * (r - g) + (r - b) * (g - b)) ** 0.5
    if b <= g:
        if den == 0:
            den = sys.float_info.min
        h = math.acos(num / den)
    elif b > g:
        if den == 0:
            den = sys.float_info.min
        h = (2 * math.pi) - math.acos(num / den)
    s = 1 - 3 * min(r, g, b) / (r + g + b)
    i = (r + b + g) / 3
    return int(h), int(s * 100), int(i * 255)


image = io.imread('wife.jpg')
hsi_image = np.zeros(image.shape, dtype='uint8')
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        r, g, b = image[i, j, :]
        h, s, i = rgb2hsi(r, g, b)
        hsi_image[i, j, :] = (h, s, i)

H = hsi_image[:, :, 0]
S = hsi_image[:, :, 1]
I = hsi_image[:, :, 2]

# 生成二值饱和度模板
S_template = np.zeros(S.shape, dtype='uint8')
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        if S[i, j] > 0.3 * S.max():
            S_template[i, j] = 1

# 色调图像与二值饱和度模板相乘可得到分割结果
F = np.zeros(H.shape, dtype='uint8')
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        F[i, j] = H[i, j] * S_template[i, j]

plt.figure()
plt.subplot(2,3,1),plt.imshow(image)
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(H,cmap ='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(S,cmap ='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(I,cmap ='gray')
plt.title('I'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(S_template)
plt.title('S_template'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(F)
plt.title('F'), plt.xticks([]), plt.yticks([])


plt.show()

