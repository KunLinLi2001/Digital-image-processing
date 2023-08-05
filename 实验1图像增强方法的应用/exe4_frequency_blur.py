'''4、对上述添加了噪声的图像综合使用频域增强方法去除图像噪声，
    并在保留图像部分低频信号的基础上强化图像的轮廓和细节。'''
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
# numpy生成高斯噪声
def gaussian(img,var=0.01):
  img = np.array(img/255, dtype=float)
  noise = np.random.normal(0,var ** 0.5,img.shape)
  out = img + noise
  out = np.clip(out, 0, 1.0)
  out = np.uint8(out*255)
  return out
# numpy生成椒盐噪声
def pepper_salt(img,prob=0.1):
    wid,len = img.shape[0],img.shape[1]
    for i in range(0,wid):
        for j in range(0,len):
            rdm = random.random()
            if rdm<prob:
                img[i][j]=0
            elif rdm>(1-prob):
                img[i][j]=255
    return img
# 巴特沃斯低通滤波器
def BLPF(img,D0,n):
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
    # 巴特沃斯低通滤波器的转移函数
    H = 1/(1+np.power(D/D0,2*n))
    # 与原图像相乘
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
# 巴特沃斯高通滤波器
def BHPF(img,D0,n):
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
    # 巴特沃斯低通滤波器的转移函数
    H = 1 / (1 + np.power(D0/ D, 2 * n))
    # 与原图像相乘
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

# 对两幅图像分别添加高斯噪声和椒盐噪声
img = cv2.imread("2.bmp",0)
noise1 = gaussian(img,0.05)
noise2 = pepper_salt(img,0.05)
# 频域增强方法去除图像噪声
res1 = BLPF(noise1,25,1)
res2 = BLPF(noise2,20,1)
# res1 = ButterworthPassFilter1(noise1,30,2)
# res2 = ButterworthPassFilter1(noise2,30,2)
# 高通滤波器强化图像的轮廓和细节
res3 = BHPF(res1,10,2)
res4 = BHPF(res2,20,2)
# 输出图像
plt.subplot(2, 3, 1), plt.imshow(noise1, cmap='gray')
plt.title('gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(res1, cmap='gray')
plt.title('gaussian->BLPF'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(res3, cmap='gray')
plt.title('gaussian->BLPF->BHPF'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(noise2, cmap='gray')
plt.title('salt&pepper'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(res2, cmap='gray')
plt.title('p&s->BLPF'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(res4, cmap='gray')
plt.title('p&s->BLPF->BHPF'), plt.xticks([]), plt.yticks([])
plt.show()







