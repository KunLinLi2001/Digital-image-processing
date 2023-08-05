from matplotlib import pyplot as plt  
import numpy as np  
import cv2  
  
# 同态滤波器  
def homomorphic(img,D0,HL,HH):  
    # 转换成浮点数类的图像矩阵，为了后续计算  
    img = np.float32(img)  
    length, width = img.shape  
    # 傅里叶变换  
    img_fft = np.fft.fft2(img)  
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
    # 保留实部  
    res = np.real(img_ifft)  
    # 转化成0-255的整型灰度值矩阵  
    res = np.uint8(np.clip(res, 0, 255))  
    return res  
  
# 读入图像  
img= cv2.imread("C:\\tmp\\exercise2_3.jpg",0)  
'''''法一：直方图修正'''  
res1 = cv2.equalizeHist(img)  
'''''法二：同态滤波'''  
res2 = homomorphic(img,10,0.5,25)  
'''''法三：添加同态滤波后再直方图修正'''  
res3 = homomorphic(img,10,0.5,2)  
res3 = cv2.equalizeHist(res3)  
'''''法四：直方图修正后再添加同态滤波'''  
res4 = homomorphic(res1,10,0.5,2)  
  
'''''展示以上方法得出的结果'''  
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 2), plt.imshow(res1, cmap='gray')  
plt.title('equalizeHist'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 3), plt.imshow(res2, cmap='gray')  
plt.title('homomorphic'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 4), plt.imshow(res3, cmap='gray')  
plt.title('homo&hist(best)'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 5), plt.imshow(res4, cmap='gray')  
plt.title('hist&homo'), plt.xticks([]), plt.yticks([])  
plt.show()  
