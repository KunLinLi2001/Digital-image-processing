import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
import random  
  
# 手写生成高斯噪声  
def gaussian(img,var=0.01):  
  img = np.array(img/255, dtype=float)  
  noise = np.random.normal(0,var ** 0.5,img.shape)  
  out = img + noise  
  out = np.clip(out, 0, 1.0)  
  out = np.uint8(out*255)  
  return out  
# 手写生成椒盐噪声  
def pepper_salt(img,prob):  
    wid,len = img.shape[0],img.shape[1]  
    for i in range(0,wid):  
        for j in range(0,len):  
            rdm = random.random()  
            if rdm<prob:  
                img[i][j]=0  
            elif rdm>(1-prob):  
                img[i][j]=255  
    return img  
# 修正后的alpha均值滤波  
def Alpha(img, d):  
    # d为需要去掉的灰度值的个数  
    model = np.ones([5, 5]) # m*n的矩阵  
    height = img.shape[0]  
    width = img.shape[1]  
    m = model.shape[0]  
    n = model.shape[1]  
    mid_x = int((m-1)/2)  
    mid_y = int((n-1)/2)  
    image_pad = np.pad(img.copy(),((mid_x,m-1-mid_x),(mid_y,n-1-mid_y)), mode="edge")  
    # 边缘填充可以防止数组下角标溢出  
    img_result = np.zeros(img.shape)  
    for i in range(height):  
        for j in range(width):  
            pad = image_pad[i:i + m,j:j + n]  
            padSort = np.sort(pad.flatten()) # 灰度值排序  
            sumAlpha = np.sum(padSort[d:m*n-d-1]) # 求和  
            img_result[i, j] = sumAlpha/(m*n-2*d) # 求均值  
    return img_result  
# 导入源图像  
img = cv2.imread("C:\\tmp\\exercise2_1.jpg", 0)  
# 图像添加高斯噪声和椒盐噪声  
noise = gaussian(img,0.01)  
noise = pepper_salt(noise,0.1)  
'''''法一：修正后的alpha均值滤波'''  
# 添加alpha改变去除灰度值的个数进行对比  
alpha1 = Alpha(noise,1)  
alpha2 = Alpha(noise,3)  
alpha3 = Alpha(noise,5)  
# alpha均值滤波输出结果  
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 2), plt.imshow(noise, cmap='gray')  
plt.title('noise'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 3), plt.imshow(alpha1, cmap='gray')  
plt.title('alpha:d=1'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 4), plt.imshow(alpha2, cmap='gray')  
plt.title('alpha:d=3'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 5), plt.imshow(alpha3, cmap='gray')  
plt.title('alpha:d=5'), plt.xticks([]), plt.yticks([])  
'''''法二：频域的巴特沃斯低通滤波器'''  
# 巴特沃斯低通滤波器,设置不同的阶数  
def BLPF(img,D0,n):  
    img = np.float32(img)  
    length, width = img.shape  
    img_fft = np.fft.fft2(img)  
    img_fftshift = np.fft.fftshift(img_fft)  
    M, N = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-length // 2, length // 2))  
    D = np.sqrt(M ** 2 + N ** 2)  
    H = 1/(1+np.power(D/D0,2*n))  
    img_fftshift = H * img_fftshift  
    img_ifftshift = np.fft.ifftshift(img_fftshift)  
    img_ifft = np.fft.ifft2(img_ifftshift)  
    res = np.real(img_ifft)  
    res = np.uint8(np.clip(res, 0, 255))  
    return res  
blpf1 = BLPF(noise,35,1)  
blpf2 = BLPF(noise,30,1)  
blpf3 = BLPF(noise,20,1)  
plt.figure(2)  
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 2), plt.imshow(noise, cmap='gray')  
plt.title('noise'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 3), plt.imshow(blpf1, cmap='gray')  
plt.title('BLPF:d0=35'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 4), plt.imshow(blpf2, cmap='gray')  
plt.title('BLPF:d0=30'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 5), plt.imshow(blpf3, cmap='gray')  
plt.title('BLPF:d0=20'), plt.xticks([]), plt.yticks([])  
plt.show()  
