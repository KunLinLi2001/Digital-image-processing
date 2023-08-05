import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread("C:\\tmp\\6.10.jpg") # 一朵花的图像
img = cv2.imread("C:\\tmp\\5.jpg") # 一朵花的图像
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgHSI = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
[R,G,B] = cv2.split(imgRGB)
[H,S,I] = cv2.split(imgHSI)

'''------------------------------1.HSI空间进行分割----------------------------------'''
'''【使用饱和度构建二值化模板】'''
threshold_S = 70 # 设定饱和度阈值为70
ret1,th_S = cv2.threshold(S,threshold_S,255,cv2.THRESH_BINARY)# 饱和度小于70置0,大于70置255
'''对色调层的高饱和区按H值进行阈值分割'''
th_H = H.copy()
# th_H = cv2.bitwise_and(th_H,th_S)
for i in range(th_H.shape[0]):
    for j in range(th_H.shape[1]):
        if th_S[i,j] == 0:
            th_H[i,j] = 255
'''对亮度层的低饱和区按I值进行阈值分割'''
th_I = I.copy()
for i in range(th_I.shape[0]):
    for j in range(th_I.shape[1]):
        if th_S[i,j] == 255:
            th_I[i,j] = 255
# th_I = cv2.bitwise_and(th_I, th_S)
'''将色调层和亮度层的分割轮廓合并到原图像的 R 层上'''
# res = cv2.add(th_I,th_H)
th_I_and_H = cv2.addWeighted(th_I,0.5,th_S,0.5,0) # 分割后色调层与亮度层合并
fix = cv2.addWeighted(th_I_and_H,0.7,R,0.3,0) # 分割轮廓合并到R层上
'''转化回彩色图像'''
res = cv2.merge([fix,G,B])
# 去除多余背景（依赖亮度）
ress = res.copy()
for i in range(th_H.shape[0]):
    for j in range(th_H.shape[1]):
        if I[i,j] <= 50:
            ress[i,j] = [0,0,0]
'''显示HSI空间进行分割的过程'''
plt.subplot(2,5,1),plt.imshow(imgRGB)
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,2),plt.imshow(R,cmap='gray')
plt.title('R'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,3),plt.imshow(th_S,cmap='gray')
plt.title('binary-S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,4),plt.imshow(th_H,cmap='gray')
plt.title('threshold-H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,5),plt.imshow(th_I,cmap='gray')
plt.title('threshold-I'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,6),plt.imshow(th_I_and_H,cmap='gray')
plt.title('th_I + th_H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,7),plt.imshow(fix,cmap='gray')
plt.title('th_I + th_H + R'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,8),plt.imshow(res)
plt.title('split-RGB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,9),plt.imshow(ress)
plt.title('del-background'), plt.xticks([]), plt.yticks([])
'''【因饱和度分割效果不是最好，考虑到亮度与图像的色彩相互独立，因此尝试使用亮度构建二值化模板】'''
threshold_I = 90
ret2,th_i = cv2.threshold(I,threshold_I,255,cv2.THRESH_BINARY)
'''对色调层的高亮度区按H值进行阈值分割'''
th_h = H.copy()
for i in range(th_h.shape[0]):
    for j in range(th_h.shape[1]):
        if th_i[i,j] == 0:
            th_h[i,j] = 255
'''对饱和度层的低亮度区按S值进行阈值分割'''
th_s = S.copy()
for i in range(th_s.shape[0]):
    for j in range(th_s.shape[1]):
        if th_i[i,j] == 255:
            th_s[i,j] = 255
# th_s = cv2.bitwise_and(th_s, th_i)
'''将色调层和饱和度层的分割轮廓合并到原图像的 R 层上'''
th_s_and_h = cv2.addWeighted(th_s,0.5,th_h,0.5,0) # 分割后色调层与亮度层合并
fix2 = cv2.addWeighted(th_s_and_h,0.7,R,0.3,0) # 分割轮廓合并到R层上
'''转化回彩色图像'''
res2 = cv2.merge([fix2,G,B])
res22 = res2.copy()
# 去除背景，只保留高亮度区域提取的图像
for i in range(th_H.shape[0]):
    for j in range(th_H.shape[1]):
        if th_i[i,j] == 0:
            res22[i,j] = [0,0,0]
'''显示HSI空间进行分割的过程'''
plt.figure(2)
plt.subplot(2,5,1),plt.imshow(imgRGB)
plt.title('Normal'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,2),plt.imshow(R,cmap='gray')
plt.title('R'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,3),plt.imshow(th_i,cmap='gray')
plt.title('binary-i'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,4),plt.imshow(th_h,cmap='gray')
plt.title('threshold-h'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,5),plt.imshow(th_s,cmap='gray')
plt.title('threshold-s'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,6),plt.imshow(th_s_and_h,cmap='gray')
plt.title('th_s + th_h'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,7),plt.imshow(fix2,cmap='gray')
plt.title('th_s + th_h + R'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,8),plt.imshow(res2)
plt.title('split-RGB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,9),plt.imshow(res22)
plt.title('del-background'), plt.xticks([]), plt.yticks([])
'''------------------------------2.RGB空间进行分割----------------------------------'''
'''【采用实验要求选取区域设置阈值】'''
# 最终遍历出x:[100,340] y:[250,560]
x1,x2 = 110,330
y1,y2 = 260,550
mask = np.zeros((x2-x1,y2-y1))
for i in range(x1,x2):
    for j in range(y1,y2):
        mask[i-x1,j-y1] = R[i,j]
plt.figure(3)
plt.subplot(1,1,1),plt.imshow(mask,cmap='gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
mean_mask = np.mean(mask) # 区域的均值
std_mask = np.std(mask) # 区域的标准差
R1 = R.copy() # R1为分割后的R分量图像
split_RGB = imgRGB.copy()
for i in range(R1.shape[0]):
    for j in range(R1.shape[1]):
        # 符合条件的像素点如下
        if R[i,j] > (mean_mask-1.25*std_mask) and R[i,j] < (mean_mask+1.25*std_mask):
            R1[i,j] = R[i,j]
        else:
            R1[i,j] = 0 # 不符合条件直接置0
            split_RGB[i,j] = [0,0,0] 
plt.figure(3)
plt.subplot(2,4,1), plt.imshow(R, cmap='gray')
plt.title('NormalR'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,2), plt.imshow(G, cmap='gray')
plt.title('NormalG'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,3), plt.imshow(B, cmap='gray')
plt.title('NormalB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,5),plt.imshow(imgRGB)
plt.title('Origin'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,6),plt.imshow(mask, cmap='gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,7),plt.imshow(R1,cmap='gray')
plt.title('split-R'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,8),plt.imshow(split_RGB)
plt.title('split-RGB'), plt.xticks([]), plt.yticks([])
'''【采用位运算掩膜分割图像】'''
# 图像颜色的限界
l = np.array([9,9,9])
h = np.array([14,14,15])
# 位运算提取图像范围中的那部分
Mask = cv2.inRange(imgRGB,l,h)
iimg = imgRGB.copy()
result = cv2.bitwise_not(iimg,iimg,mask=Mask)
plt.figure(4)
plt.subplot(1,2,1),plt.imshow(imgRGB)
plt.title('Origin'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result)
plt.title('split'), plt.xticks([]), plt.yticks([])

plt.show()
