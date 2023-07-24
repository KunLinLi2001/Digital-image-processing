import numpy as np
import cv2

img = cv2.imread('wife.jpg',0)
cv2.imshow("origin",img)
D0,HL,HH = 10,0.5,2
# D0 = float(input("输入截止频率："))
# HL = float(input("输入最低转换值："))
# HH = float(input("输入最高转换值："))
# 转换成浮点数类的图像矩阵，为了后续计算
# img = np.float32(img)
length, width = img.shape
# 傅里叶变换
img_fft = np.fft.fft2(img)
# img_fft = cv2.dft(img)

# 零频率分量移到频谱中心
img_fftshift = np.fft.fftshift(img_fft)
# 所有像素点坐标的坐标矩阵为M和N
M, N = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-length // 2, length // 2))
# 每个坐标的频率计算出来
D = np.sqrt(M ** 2 + N ** 2)
# 同态滤波器的转移函数
H = (HH - HL) * (1 - np.exp(-(D ** 2 / D0 ** 2))) + HL
# 加上同态滤波
img_fftshift = H * img_fftshift
# 将频域中心移回原位置
img_ifftshift = np.fft.ifftshift(img_fftshift)
# 傅里叶逆变换
img_ifft = np.fft.ifft2(img_ifftshift)
# img_ifft = cv2.idft(img_ifftshift)

# 保留实部
res = np.real(img_ifft)
# 转化成0-255的整型灰度值矩阵
res = np.uint8(np.clip(res, 0, 255))

cv2.imshow("now",res)
cv2.waitKey(0)
