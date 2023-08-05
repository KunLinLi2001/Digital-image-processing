import cv2
import matplotlib.pyplot as plt
import numpy as np

def calcAndDrawHist(image):
    b, g, r = cv2.split(image)

    # 显示图像
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_title('b hist')
    ax[0, 0].hist(b.ravel(), bins=256)
    ax[0, 1].set_title('g hist')
    ax[0, 1].hist(g.ravel(), bins=256)
    ax[1, 0].set_title('r hist')
    ax[1, 0].hist(r.ravel(), bins=256)
    ax[1, 1].set_title('src')
    ax[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');
    ax[1, 1].axis('off')  # 关闭坐标轴显示
    plt.show()

def RGB_separate(image, l_b, l_g, l_r, h_b, h_g, h_r):
    # 定义提取颜色的上下限
    lower = np.array([l_b, l_g, l_r])
    higher = np.array([h_b, h_g, h_r])
    # 提取图像中界限范围的部分
    mask = cv2.inRange(image, lower, higher)
    left = cv2.bitwise_not(img, img, mask=mask)
    return left

def HSI_separete(image, thresh):
    # value 局部阈值
    # 根据图像的hsi，得到该图像i分量的影响较大，固该函数以亮度为模板进行掩膜操作
    img_hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # HSI空间通道0是色调，1是饱和度， 2是亮度
    h = img_hsi[:, :, 0]
    s = img_hsi[:, :, 1]
    i = img_hsi[:, :, 2]
    # threshold(InputArray src, OutputArray dst, double thres, double maxval, int type)
    # src：源图像，可以为8位的灰度图，也可以为32位的彩色图像；
    # dst：输出图像；
    # thresh：阈值；
    # maxval：二值图像中灰度最大值
    # type：阈值操作类型
    ret_i, binary_i = cv2.threshold(i, thresh, 255, cv2.THRESH_BINARY)
    # 与运算
    h = cv2.bitwise_and(h, binary_i)
    s = cv2.bitwise_and(s, binary_i)
    i = cv2.bitwise_and(i, binary_i)
    img_hsi = cv2.merge([h, s, i])
    image_rgb = cv2.cvtColor(img_hsi, cv2.COLOR_HSV2BGR)
    return image_rgb

img = cv2.imread("C:\\tmp\\6.10.jpg")
rgb_show = calcAndDrawHist(img)
HSIImg = HSI_separete(img, 100)
RGBImg = RGB_separate(img, l_b=9, l_g=9, l_r=9, h_b=13, h_g=13, h_r=14)
cv2.imshow("HSI_separete", HSIImg)
cv2.imshow("RGB_separete", RGBImg)
cv2.waitKey()
cv2.destroyAllWindows()
