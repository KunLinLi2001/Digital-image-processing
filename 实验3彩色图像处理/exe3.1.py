import cv2
import numpy as np
from matplotlib import pyplot as plt

# 手写RGB转HSI函数
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
    R_G_B = R+G+B+esp # 三层相加
    # R_G_B = np.where(R_G_B<=0,1e-8,R_G_B)
    # 找到Min,计算对应像素点的S
    Min = np.where(B>=G,G,B)
    Min = np.where(Min<=R,Min,R)
    S = 1-(3/(R_G_B))*Min
    '''计算H'''
    # 求解夹角：分母添加1e-8是避免分母为0的情况
    angle = np.arccos((0.5*(R+R-G-B))/(np.sqrt((R-G)*(R-G)+(R-B)*(G-B)))+esp)
    
    # for i in range(0,wid):
    #     for j in range(0,len):
    #         if(B[i,j]>G[i,j]):
    #             H[i,j] = 2*np.pi-angle[i][j]
    #         else:
    #             H[i,j] =angle[i][j]
    #         H[i,j] =H[i,j]/(2*np.pi)
    #         if(S[i,j]==0):
    #             H[i,j] =0
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

def RGB222HSI(img):
    #保存原始图像的行列数
    row,col = img.shape[0],img.shape[1]
    #对原始图像进行复制
    hsi_img = img.copy()
    #对图像进行通道拆分
    B,G,R = cv2.split(img)
    #把通道归一化到[0,1]
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))    #定义H通道
    I = (R + G + B) / 3.0       #计算I通道
    S = np.zeros((row,col))      #定义S通道
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #计算夹角
        h = np.zeros(col)               #定义临时数组
        #den>0且G>=B的元素h赋值为thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        #den>0且G<=B的元素h赋值为thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        #den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      #弧度化后赋值给H通道
    #计算S通道
    for i in range(row):
        min = []
        #找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        #计算S通道
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        #I为0的值直接赋值0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    #扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img,H*255,S*255,I*255

def RGB22HSI(img1):
    img1 = img1.astype('float32')
    b, g, r = img1[:, :, 0]/255.0, img1[:, :, 1]/255.0, img1[:, :, 2]/255.0

    I = (r+g+b)/3.0

    tem = np.where(b >= g, g, b)
    minValue = np.where(tem >= r, r, tem)
    S = 1 - (3 / (r + g + b)) * minValue

    num1 = 2*r - g - b
    num2 = 2*np.sqrt(((r - g) ** 2) + (r-b)*(g-b))
    deg = np.arccos(num1/num2)
    H = np.where(g >= b, deg, 2*np.pi - deg)

    resImg = np.zeros((img1.shape[0], img1.shape[1],img1.shape[2]), dtype=np.float)
    resImg[:, :, 0], resImg[:, :, 1], resImg[:, :, 2] = H*255, S*255, I*255
    resImg = resImg.astype('uint8')
    return resImg,H*255,S*255,I*255



# 读入彩色图像(BGR排布)
img = cv2.imread("C:\\tmp\\6.14.jpg") # 两朵花的图像
(b,g,r) = cv2.split(img)
# 转化成RGB排布的彩色图像
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
(R,G,B) = cv2.split(img)

'''分别显示RGB空间的各层'''
plt.figure(1)
plt.subplot(2,3,1),plt.imshow(imgRGB)
plt.title('Normal'), plt.xticks([]), plt.yticks([])
# Error是为了查看误将BGR图像作为RGB图像的后果
plt.subplot(2,3,4),plt.imshow(img)
plt.title('Error'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(R,cmap ='gray')
plt.title('NormalR'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(G,cmap ='gray')
plt.title('NormalG'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(B,cmap ='gray')
plt.title('NormalB'), plt.xticks([]), plt.yticks([])
# ErrorR是为了查看将单图层图像作为三图层不加cmap约束后果
plt.subplot(2,3,6),plt.imshow(R)
plt.title('ErrorR'), plt.xticks([]), plt.yticks([])

'''将RGB图像转换到HSI空间'''
imgHSI,H,S,I = RGB2HSI(imgRGB)

'''显示HSI空间的各层'''
plt.figure(2)
plt.subplot(2,2,1),plt.imshow(imgHSI)
plt.title('HSI'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(H,cmap ='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(S,cmap ='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(I,cmap ='gray')
plt.title('I'), plt.xticks([]), plt.yticks([])
'''调用库函数转换HSV模型，检验自身所写的转换HSI代码效果方向正确'''
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
[h,s,v] = cv2.split(imgHSV)
'''显示HSV空间的各层'''
plt.figure(3)
plt.subplot(2,2,1),plt.imshow(imgHSV)
plt.title('HSV'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(h,cmap ='gray')
plt.title('H'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(s,cmap ='gray')
plt.title('S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(v,cmap ='gray')
plt.title('V'), plt.xticks([]), plt.yticks([])
plt.show()