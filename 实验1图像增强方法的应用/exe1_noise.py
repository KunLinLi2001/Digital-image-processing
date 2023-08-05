'''
使用numpy中的函数给图像1分别添加椒盐噪声和高斯噪声，给图像2同时添加椒盐噪声和高斯噪声。
'''
import random
import cv2
import skimage
import numpy as np
from matplotlib import pyplot as plt
# skimage 生成高斯噪声
def gaussian1(img,var=0.01):
    noise = skimage.util.random_noise(img,mode='gaussian',var=0.01)
    noise = noise*255
    noise = noise.astype(np.uint8)
    return noise
# 手写生成高斯噪声
def gaussian2(img,var=0.01):
  img = np.array(img/255, dtype=float)
  noise = np.random.normal(0,var ** 0.5,img.shape)
  out = img + noise
  out = np.clip(out, 0, 1.0)
  out = np.uint8(out*255)
  return out
# skimage 生成椒盐噪声
def pepper_salt1(img,prob):
    noise = skimage.util.random_noise(img, mode='s&p', salt_vs_pepper=prob)
    noise = noise * 255
    noise = noise.astype(np.uint8)
    return noise
# 手写生成椒盐噪声
def pepper_salt2(img,prob):
    wid,len = img.shape[0],img.shape[1]
    for i in range(0,wid):
        for j in range(0,len):
            rdm = random.random()
            if rdm<prob:
                img[i][j]=0
            elif rdm>(1-prob):
                img[i][j]=255
    return img
# 运行该文件时，才进行下列输出操作；如果只是把exe1当做一个包，就不进行下列操作
if __name__ == '__main__':
    img1 = cv2.imread("C:\\tmp\\1.jpg",0)
    # 对图像单独添加高斯噪声/椒盐噪声，对比一下库函数和手写函数的区别
    noise1 = gaussian1(img1,0.01)
    noise2 = pepper_salt1(img1,0.5)
    noise3 = gaussian2(img1,0.01)
    noise4 = pepper_salt2(img1,0.04)
    # figure1展示对图像1分别添加椒盐噪声和高斯噪声
    plt.figure(1)
    plt.subplot(2, 2, 1), plt.imshow(noise1, cmap='gray')
    plt.title('gaussian by skimage'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(noise2, cmap='gray')
    plt.title('pepper&salt by skimage'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(noise3, cmap='gray')
    plt.title('gaussian by numpy'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(noise4, cmap='gray')
    plt.title('pepper&salt by numpy'), plt.xticks([]), plt.yticks([])
    # 对图像2同时添加两种噪声
    img2 = cv2.imread("2.bmp",0)
    noise11 = gaussian1(img2,0.01)
    noise11 = pepper_salt1(noise11,0.5)
    noise22 = gaussian2(img2,0.01)
    noise22 = pepper_salt2(noise22,0.04)
    # figure2展示给图像2同时添加椒盐噪声和高斯噪声
    plt.figure(2)
    plt.subplot(1, 2, 1), plt.imshow(noise11, cmap='gray')
    plt.title('fixed noise by skimage'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(noise22, cmap='gray')
    plt.title('fixed noise by numpy'), plt.xticks([]), plt.yticks([])
    plt.show()
