from matplotlib import pyplot as plt  
import numpy as np  
from numpy import fft  
import math  
import cv2  
  
# 手写生成高斯噪声  
def gaussian(img,var=0.01):  
  img = np.array(img/255, dtype=float)  
  noise = np.random.normal(0,var ** 0.5,img.shape)  
  out = img + noise  
  out = np.clip(out, 0, 1.0)  
  out = np.uint8(out*255)  
  return out  
# 仿真运动模糊，构建运动模糊的PSF  
def motion_process(img_size, motion_angle):  
    PSF = np.zeros(img_size)  
    print(img_size)  
    center_position = (img_size[0] - 1) / 2  
    slope_tan = math.tan(motion_angle * math.pi / 180)  
    slope_cot = 1 / slope_tan  
    if slope_tan<=1:  
        for i in range(15):  
            offset = round(i*slope_tan)  
            PSF[int(center_position+offset),int(center_position-offset)] = 1  
        return PSF/PSF.sum()  # 对点扩散函数进行归一化亮度  
    else:  
        for i in range(15):  
            offset = round(i * slope_cot)  
            PSF[int(center_position - offset), int(center_position + offset)] = 1  
        return PSF / PSF.sum()  
# 对图片进行运动模糊  
def motion(input, PSF, eps):  
    input_fft = fft.fft2(input)  # 傅里叶变换  
    PSF_fft = fft.fft2(PSF) + eps # 添加平均噪声功率混合  
    out = fft.ifft2(input_fft * PSF_fft) # 相当于时域的卷积处理  
    out = np.abs(fft.fftshift(out)) # 取出实部  
    return out  
# 图像添加逆滤波（设置禁止频率的改进版本）  
# def inverse(img, H, w0):  
#     img_h, img_w = img.shape[0],img.shape[1] 
#     eps = 1e-3 
#     src_fft = np.fft.fft2(img) # 傅里叶变换
#     # src_fft = np.fft.fftshift(src_fft)    
#     H_fft = np.fft.fft2(H) + eps # 添加平均噪声功率混合 eps=1e-6  
#     # H_fft = np.fft.fftshift(H_fft)
#     M = np.zeros(src_fft.shape)  
#     '''''遍历图像进行频率的限制'''  
#     for i in range(-img_h//2,img_h//2):  
#         for j in range(-img_w//2,img_w//2):   
#             if (i**2 + j**2) <= w0**2:  
#                 M[i, j] = 1
#             else:  
#                 M[i, j] = np.abs(1/H_fft[i, j])
#     # out = np.fft.ifftshift(src_fft*M/H_fft) # 将中心点恢复！！！这句话很关键！  
#     # out = np.fft.ifftshift(src_fft/H_fft) # 将中心点恢复！！！这句话很关键！  
#     # out = np.fft.ifft2(out) # 傅里叶反变换  
#     out = np.fft.ifft2(src_fft*/H_fft) # 傅里叶反变换 
#     out = np.abs(out) # 取出实部  
#     return out  

def inverse(img,H,w0):
    eps = 1e-3
    M,N = img.shape[1],img.shape[0] 
    u,v = np.meshgrid(range(0,M),range(0,N))
    '''傅里叶变换+原点移至中心'''
    img = img.astype(np.float32)
    src_fft = np.fft.fft2(img) # 傅里叶变换
    src_fft = np.fft.fftshift(src_fft) # 将低频分量移动到频域图像中心
    H = H.astype(np.float32)
    H_FFT = np.fft.fft2(H)+eps # 傅里叶变换
    H_FFT = np.fft.fftshift(H_FFT) # 将低频分量移动到频域图像中心
    '''在频域率设置截止频率'''
    if(w0==0):
        out = src_fft/H_FFT
    else:
        D = np.sqrt((u - M//2)**2 + (v - N//2)**2)
        kernel = np.zeros(img.shape[:2], np.float32)
        kernel[D<=w0] = 1
        out = src_fft/H_FFT*kernel
    '''傅里叶逆变换，逆中心化'''
    out = np.fft.ifft2(out)  # 逆傅里叶变换，返回值是复数数组
    out = np.fft.ifftshift(out) # 将低频分量逆转换回图像四角
    out = np.abs(out)
    out = np.uint8(cv2.normalize(out,None,0,255,cv2.NORM_MINMAX)) # 归一化为 [0,255]
    return out


# def inverse(input, PSF, eps):       # 逆滤波
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps #噪声功率，这是已知的，考虑epsilon
#     result = fft.ifft2(input_fft / PSF_fft) #计算F(u,v)的傅里叶反变换
#     result = np.abs(fft.fftshift(result))
#     return result


# 图像添加维纳滤波  
def wiener(input, PSF, eps, K=0.01):  
    FFT = fft.fft2(input)  # 傅里叶变换  
    PSF_fft = fft.fft2(PSF) + eps  # 添加平均噪声功率混合 eps=1e-3  
    PSF_fft_1 = np.conj(PSF_fft)/(np.abs(PSF_fft) ** 2 + K)  # 计算F(u,v)的傅里叶反变换  
    out= fft.ifft2(FFT*PSF_fft_1)  
    out = fft.fftshift(out) # 将零频率分量移到图像的中心  
    out = np.abs(out)  # 取出实部  
    return out  
# 导入源图像  
img = cv2.imread("C:\\tmp\\exercise2_2.png",0)  
img_h,img_w = img.shape[0],img.shape[1]  
# 图像添加高斯噪声和运动模糊  
noise = gaussian(img,0.005)  
PSF = motion_process((img_h, img_w), 200)  
noise = np.abs(motion(noise,PSF,1e-3))  
'''''方法一：添加逆滤波'''  
# 添加逆滤波（不同的截止频率）  
inverse1 = inverse(noise,PSF,0)  
inverse2 = inverse(noise,PSF,40)  
inverse3 = inverse(noise,PSF,100)  
inverse4 = inverse(noise,PSF,120)  
# 添加逆滤波后的结果  
plt.figure(1)  
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 2), plt.imshow(noise, cmap='gray')  
plt.title('add motion&noise'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 3), plt.imshow(inverse1, cmap='gray')  
plt.title('inverse:w0=0'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 4), plt.imshow(inverse2, cmap='gray')  
plt.title('inverse:w0=40'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 5), plt.imshow(inverse3, cmap='gray')  
plt.title('inverse:w0=100'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 6), plt.imshow(inverse4, cmap='gray')  
plt.title('inverse:w0=120'), plt.xticks([]), plt.yticks([])  
'''''方法二：添加维纳滤波'''  
# 添加维纳滤波  
wiener1 = wiener(noise,PSF,1e-3,10)  
wiener2 = wiener(noise,PSF,1e-3,1)  
wiener3 = wiener(noise,PSF,1e-3,0.01)  
wiener4 = wiener(noise,PSF,1e-1,0.00001)  
# 输出维纳滤波的结果  
plt.figure(2)  
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')  
plt.title('origin'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 2), plt.imshow(noise, cmap='gray')  
plt.title('add motion&noise'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 3), plt.imshow(wiener1, cmap='gray')  
plt.title('wiener:k=10'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 4), plt.imshow(wiener2, cmap='gray')  
plt.title('wiener:k=1'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 5), plt.imshow(wiener3, cmap='gray')  
plt.title('wiener:k=0.01(best)'), plt.xticks([]), plt.yticks([])  
plt.subplot(2, 3, 6), plt.imshow(wiener4, cmap='gray')  
plt.title('wiener:k=0.00001'), plt.xticks([]), plt.yticks([])  
plt.show()  





