# 2.给彩色图像添加椒盐噪声，使用中值滤波对彩色图像进行去噪，去噪后分别在RGB空间和HSI空间进行直方图均衡化和图像锐化，比较两个空间的增强效果
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from math import cos

# 手写生成椒盐噪声
def pepper_salt(img,prob):
    wid,len = img.shape[0],img.shape[1]
    for i in range(0,wid):
        for j in range(0,len):
            rdm = random.random()
            if rdm < prob:
                img[i][j] = 0
            elif rdm > (1-prob):
                img[i][j] = 255
    return img
# 手写RGB转HSI
def RGB2HSI(img):
    esp = 1e-8
    # 为了计算转换成float32类型
    img = img.astype('float32')
    (R,G,B) = cv2.split(img)
    wid,len = img.shape[0],img.shape[1]
    # R,G,B = R/255,G/255,B/255
    '''计算I'''
    I = (R+G+B)/3
    '''计算S'''
    # 解决计算S时分母出现0的问题
    R_G_B = R+G+B # 三层相加
    # R_G_B = np.where(R_G_B<=0,1e-8,R_G_B)
    # 找到Min,计算对应像素点的S
    Min = np.where(B>=G,G,B)
    Min = np.where(Min<=R,Min,R)
    S = 1-(3/(R_G_B+esp))*Min
    '''计算H'''
    # 求解夹角：分母添加1e-8是避免分母为0的情况
    angle = np.arccos((0.5*(R+R-G-B))/(np.sqrt((R-G)*(R-G)+(R-B)*(G-B))+esp))
    H = np.where(B>G,2*np.pi-angle,angle)
    H = H/(2*np.pi)
    H = np.where(S==0,0,H)
    # H = np.uint8(H*255)
    # S = np.uint8(S*255)
    # I = np.uint8(I*255)
    imgHSI = img.copy()
    H = cv2.normalize(H,None,0,1,cv2.NORM_MINMAX)
    S = cv2.normalize(S,None,0,1,cv2.NORM_MINMAX)
    I = cv2.normalize(I,None,0,1,cv2.NORM_MINMAX)
    imgHSI[:,:,0] = H*255
    imgHSI[:,:,1] = S*255
    imgHSI[:,:,2] = I*255
    imgHSI = np.uint8(imgHSI)
    return [imgHSI,H*255,S*255,I*255]
# 手写HSI转RGB
def HSI2RGB(img):
    eps = 1e-6
    img = (img).astype('float64') / 255.0
    for k in range(img.shape[0]):
        for j in range(img.shape[1]):
            h, s, i = img[k, j, 0], img[k, j, 1], img[k, j, 2]
            r, g, b = 0, 0, 0
            if 0 <= h < 2/3*np.pi:
                b = i * (1 - s)
                r = i * (1 + s * cos(h) / (cos(np.pi/3-h)+eps)) # 避免除0
                g = 3 * i - (b + r)
            elif 2/3*np.pi <= h < 4/3*np.pi:
                r = i * (1 - s)
                g = i * (1 + s * cos(h-2/3*np.pi) / (cos(np.pi - h)+eps))
                b = 3 * i - (r + g)
            elif 4/3*np.pi <= h <= 5/3*np.pi:
                g = i * (1 - s)
                b = i * (1 + s * cos(h - 4/3*np.pi) / (cos(5/3*np.pi - h)+eps))
                r = 3 * i - (g + b)
            img[k, j, 0], img[k, j, 1], img[k, j, 2] = r, g, b
    return (img * 255).astype('uint8')

'''显示原RGB图像的效果'''
img = cv2.imread("C:\\tmp\\4.9.bmp") # 山峰图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.subplot(3, 2, 1), plt.imshow(img)
plt.title('origin'), plt.xticks([]), plt.yticks([])
'''对彩色图像添加椒盐噪声，使用中值滤波对整个彩色图像进行去噪'''
noise = pepper_salt(img,0.02)
clear1 = cv2.medianBlur(noise,ksize=3) #直接对整个彩色图像使用中值滤波
clear2 = cv2.medianBlur(noise,ksize=7)
plt.subplot(3, 2, 2), plt.imshow(noise)
plt.title('add-noise'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 3), plt.imshow(clear1)
plt.title('all-clear(size=3)'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 4), plt.imshow(clear2)
plt.title('all-clear(size=7)'), plt.xticks([]), plt.yticks([])
'''分离三个色层，分别添加中值滤波'''
[R,G,B] = cv2.split(noise) # 提取三个色层
R_clear1 = cv2.medianBlur(R,ksize=3)
G_clear1 = cv2.medianBlur(G,ksize=3)
B_clear1 = cv2.medianBlur(B,ksize=3)
R_clear2 = cv2.medianBlur(R,ksize=5)
G_clear2 = cv2.medianBlur(G,ksize=5)
B_clear2 = cv2.medianBlur(B,ksize=5)
clear3 = cv2.merge([R_clear1,G_clear1,B_clear1])
clear4 = cv2.merge([R_clear2,G_clear2,B_clear2])
plt.subplot(3, 2, 5), plt.imshow(clear3)
plt.title('split-clear(size=3)'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 6), plt.imshow(clear4)
plt.title('split-clear(size=5)'), plt.xticks([]), plt.yticks([])
'''在RGB空间进行直方图均衡化'''
hist_r1 = cv2.equalizeHist(R_clear1)
hist_g1 = cv2.equalizeHist(G_clear1)
hist_b1 = cv2.equalizeHist(B_clear1)
hist1 = cv2.merge([hist_r1,hist_g1,hist_b1])
plt.figure(2)
plt.subplot(3, 3, 1), plt.imshow(clear3) # clear3是三个图层均添加中值滤波去除噪声的图像
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(hist1)
plt.title('equalizeHist'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3) # 均衡化后的彩色直方图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([hist1], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
plt.title('equalizeHist'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(hist_r1,cmap='gray')
plt.title('equalizeHist-r'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(hist_g1,cmap='gray')
plt.title('equalizeHist-g'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(hist_b1,cmap='gray')
plt.title('equalizeHist-b'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(R_clear1,cmap='gray')
plt.title('origin-r'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(G_clear1,cmap='gray')
plt.title('origin-g'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(B_clear1,cmap='gray')
plt.title('origin-b'), plt.xticks([]), plt.yticks([])
'''在RGB空间进行图像锐化——拉普拉斯算子'''
la = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)# 拉普拉斯算子
# laplacian1 = cv2.filter2D(clear3,-1,kernel=la)# 利用卷积进行锐化
la_r1 = cv2.filter2D(R_clear1,-1,kernel=la)# 利用卷积进行锐化
la_g1 = cv2.filter2D(G_clear1,-1,kernel=la)
la_b1 = cv2.filter2D(B_clear1,-1,kernel=la)
la1 = cv2.merge([la_r1,la_g1,la_b1])
plt.figure(3)
plt.subplot(3, 3, 1), plt.imshow(clear3) # clear3是三个图层均添加中值滤波去除噪声的图像
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(la1)
plt.title('laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(la_r1,cmap='gray')
plt.title('la-r'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(la_g1,cmap='gray')
plt.title('la-g'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(la_b1,cmap='gray')
plt.title('la-b'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(R_clear1,cmap='gray')
plt.title('origin-r'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(G_clear1,cmap='gray')
plt.title('origin-g'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(B_clear1,cmap='gray')
plt.title('origin-b'), plt.xticks([]), plt.yticks([])
'''---------------在HSI空间进行直方图均衡化和图像锐化（仅需处理亮度I）---------------'''
imgHSI = cv2.cvtColor(clear3, cv2.COLOR_RGB2HSV)
[H,S,I] = cv2.split(imgHSI)
I = I.astype('uint8')
histI = cv2.equalizeHist(I) # 对I进行直方图均衡化
laI = cv2.filter2D(I,-1,kernel=la) # 对I添加拉普拉斯算子进行锐化
histHSI,laHSI = clear3.copy(),clear3.copy()
histHSI[:,:,0],histHSI[:,:,1],histHSI[:,:,2] = H,S,histI # 构建直方图修正后的彩色图像
laHSI[:,:,0],laHSI[:,:,1],laHSI[:,:,2] = H,S,laI # 构建锐化后修正后的彩色图像
hist2 = cv2.cvtColor(histHSI, cv2.COLOR_HSV2RGB)
la2 = cv2.cvtColor(laHSI, cv2.COLOR_HSV2RGB)
plt.figure(4)
plt.subplot(1, 3, 1), plt.imshow(clear3) # clear3是三个图层均添加中值滤波去除噪声的图像
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(hist2)
plt.title('equalizeHist-I'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(la2)
plt.title('laplacian-I'), plt.xticks([]), plt.yticks([])
'''比较两种锐化方式'''
la1gray,la2gray,com = H.copy(),H.copy(),H.copy()
la1gray,la2gray = cv2.cvtColor(la1, cv2.COLOR_RGB2GRAY),cv2.cvtColor(la2, cv2.COLOR_RGB2GRAY)
com = np.abs(la1gray-la2gray)
plt.figure(5)
plt.subplot(2, 2, 1), plt.imshow(clear3) # clear3是三个图层均添加中值滤波去除噪声的图像
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(la1)
plt.title('la-GRB'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(la2)
plt.title('la-I'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(com,cmap='gray')
plt.title('compare'), plt.xticks([]), plt.yticks([])
plt.show()
