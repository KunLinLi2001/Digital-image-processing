import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def draw_gray_histogram(img_gray):

    img_rgb = cv.cvtColor(img_gray, cv.COLOR_BGR2RGB)

    print(img_gray.ravel())
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.subplot(122)
    plt.hist(img_gray.ravel(), 256, [0, 255])
    plt.show()

def draw_color_histogram(img_color):
    color_bgr = ('b', 'g', 'r')
    plt.subplot(121)
    plt.imshow(img_color)
    for i, j in enumerate(color_bgr):

        hist = cv.calcHist([img_color], [i], None, [256], [0, 256])
        plt.subplot(122)
        plt.plot(hist, color=j)
    plt.show()

def draw_hasmask_histogram(img_tomask):
    # ���쵲��
    mask = np.zeros(img_tomask.shape[:2], np.uint8)
    mask[60:160, 60:160] = 1
    # �����
    img_has_mask = cv.bitwise_and(img_tomask, img_tomask, mask=mask)
    # ����ԭͼ�����������ֱ��ͼ
    hist_O = cv.calcHist([img_tomask], [0], None, [256], [0, 256])
    hist_M = cv.calcHist([img_tomask], [0], mask, [256], [0, 256])
    plt.subplot(121)
    plt.imshow(img_tomask)
    plt.subplot(122)
    plt.plot(hist_O)
    plt.show()
    plt.subplot(121)
    plt.imshow(img_has_mask)
    plt.subplot(122)
    plt.plot(hist_M)
    plt.show()
if __name__ == '__main__':
    img = cv.imread(r"F:\temp\A_6655.jpg")
    draw_color_histogram(img)

