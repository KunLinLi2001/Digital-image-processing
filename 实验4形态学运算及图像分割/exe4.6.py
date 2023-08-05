'''对车牌图像分别利用基本的全局阈值处理方法和 Otsu 算法法进行图像中汽车及车牌的分割，
显示处理前、后图像；思考不同的阈值处理算法对分割效果的影响？'''
import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread("C:\\tmp\\four\\4.6.jpg",0)
img2 = cv2.imread("C:\\tmp\\four\\4.66.jpg",0)
'''全局阈值处理（利用迭代法进行循环处理）'''
def Global_threshold_processing(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 构建直方图
    T = round(img.mean()) # 计算图像灰度的中值作为初始阈值T
    num = img.shape[0]*img.shape[1] # 计算像素点的个数，为计算阈值两侧均值做准备
    while True:
        num_l,num_r,sum_l,sum_r = 0,0,0,0 # 直方图中阈值T两侧像素点的个数
        for i in range(0,T):
            num_l += hist[i,0] # 阈值左侧像素数量
            sum_l += hist[i,0]*i
        for i in range(T,256):
            num_r += hist[i,0] # 阈值左侧像素数量
            sum_r += hist[i,0]*i
        T1 = round(sum_l/num_l) # 左端平均灰度值
        T2 = round(sum_r/num_r) # 右端平均灰度值
        tnew = round((T1+T2)/2) # 计算出的新阈值
        if T==tnew: # 两侧的平均灰度值相同
            break
        else:
            T = tnew # 更新阈值
    ret,dst = cv2.threshold(img,T,255,cv2.THRESH_BINARY) # 阈值分割
    return dst

# '''Otsu 算法'''
# def otsu(img):
#     hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
#     total = img.shape[0] * img.shape[1]  # 像素总数
#     Mean = img.mean() # 图像的平均灰度
#     gt = np.zeros(256) # g(t)就是当分割阈值为t时的类间方差表达式
#     num_ft,sum_ft = 0,0
#     for t in range(256):
res = Global_threshold_processing(img2)
cv2.imshow("aa",res)
cv2.waitKey(0)

#     return out

#     # 11.16 图像分割之全局阈值处理
# def fun(img):
#     eps = 1  # 给定的值，循环直到灰度值差别小于eps
#     histCV = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
#     print(histCV)
#     grayScale = range(256)  # 灰度级 [0,255]
#     totalPixels = img.shape[0] * img.shape[1]  # 像素总数
#     totalGary = np.dot(histCV[:,0], grayScale)  # 内积, 总和灰度值
#     T = round(totalGary/totalPixels)  # 平均灰度

#     while True:
#         numC1, sumC1 = 0, 0
#         for i in range(T): # 计算 C1: (0,T) 平均灰度
#             numC1 += histCV[i,0]  # C1 像素数量
#             sumC1 += histCV[i,0] * i  # C1 灰度值总和
#         numC2, sumC2 = (totalPixels-numC1), (totalGary-sumC1)  # C2 像素数量, 灰度值总和
#         T1 = round(sumC1/numC1)  # C1 平均灰度
#         T2 = round(sumC2/numC2)  # C2 平均灰度
#         Tnew = round((T1+T2)/2)  # 计算新的阈值
#         print("T={}, m1={}, m2={}, Tnew={}".format(T, T1, T2, Tnew))
#         if abs(T-Tnew) < eps:  # 等价于 T==Tnew
#             break
#         else:
#             T = Tnew

#     # 阈值处理
#     ret, imgBin = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)  # 阈值分割, thresh=T

#     plt.figure(figsize=(11, 4))
#     plt.subplot(131), plt.title("original"), plt.imshow(img, 'gray')
#     plt.subplot(132, yticks=[]), plt.title("Gray Hist")  # 直方图
#     histNP, bins = np.histogram(img.flatten(), bins=255, range=[0, 255], density=True)
#     plt.bar(bins[:-1], histNP[:])
#     plt.subplot(133), plt.title("threshold={}".format(T)), plt.axis('off')
#     plt.imshow(imgBin, 'gray')
#     plt.tight_layout()
#     plt.show()

# fun(img1)