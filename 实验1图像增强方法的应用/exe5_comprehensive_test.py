'''对上述添加了噪声的图像，综合运用空间域增强方法和频率域图像增强方法，
   实现图像的滤除噪声，强化显示图像的边缘和细节的效果。
'''
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
'''阶段一：读入灰度图，在图像上同时添加高斯噪声和椒盐噪声'''
img = cv2.imread("2.bmp",0)
noise = gaussian(img,0.01)
noise = pepper_salt(noise,0.05)
plt.figure(1) # figure1显示原图像与添加噪声的图像
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(noise, cmap='gray')
plt.title('fixed noise'), plt.xticks([]), plt.yticks([])
'''阶段二：对添加噪声的图像进行空域和频率域的平滑去噪'''
res1 = cv2.blur(noise,(4,4))# 添加均值滤波
res2 = cv2.medianBlur(res1,7)# 在添加均值滤波基础上加中值滤波
res3 = BLPF(res2,35,1)# 空域处理的基础上添加BLPF
plt.figure(2) # figure2显示图像平滑后的效果
plt.subplot(1, 3, 1), plt.imshow(res1, cmap='gray')
plt.title('noise->average'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(res2, cmap='gray')
plt.title('noise->ave->mid'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(res3, cmap='gray')
plt.title('noise->ave->mid->BLPF'), plt.xticks([]), plt.yticks([])
'''阶段三：对平滑后的图像进行锐化提取细节特征'''
res4 = cv2.Sobel(res3,cv2.CV_64F,1,0,ksize=5)# 仅添加sobel算子锐化
res5 = BHPF(res3,20,1) # 仅添加巴特沃斯高通滤波器
res6 = BHPF(res4,20,1) # res4基础上添加BHPF
plt.figure(3) # figure3显示图像锐化后的效果
plt.subplot(1, 3, 1), plt.imshow(res4, cmap='gray')
plt.title('noise->pinghua->sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(res5, cmap='gray')
plt.title('noise->pinghua->BHPF'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(res6, cmap='gray')
plt.title('noise->pinghua->sobel->BHPF'), plt.xticks([]), plt.yticks([])
plt.show()
















