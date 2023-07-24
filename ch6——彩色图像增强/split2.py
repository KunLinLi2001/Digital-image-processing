from skimage import data, color, io
from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

image = cv2.imread(r'wife.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]
# RGB颜色空间中的分割
# 选择样本区域
r_template = r[128:255, 85:169]
# 计算该区域的彩色点的平均向量a的红色分量
r_template_u = np.mean(r_template)
# 计算样本点红色分量的标准差
r_template_d = 0.0
for i in range(r_template.shape[0]):
    for j in range(r_template.shape[1]):
        r_template_d = r_template_d + (r_template[i, j] - r_template_u) * (r_template[i, j] - r_template_u)

r_template_d = math.sqrt(r_template_d / r_template.shape[0] / r_template.shape[1])
# 寻找符合条件的点，r_cut为红色分割图像
r_cut = np.zeros(r.shape, dtype='uint8')
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r[i, j] >= (r_template_u - 1.25 * r_template_d) and r[i, j] <= (r_template_u + 1.25 * r_template_d):
            r_cut[i, j] = 1
# image_cut为根据红色分割后的RGB图像
image_cut = np.zeros(image.shape, dtype='uint8')
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r_cut[i, j] == 1:
            image_cut[i, j, :] = image[i, j, :]

plt.figure()
plt.subplot(2,3,1),plt.imshow(image)
plt.title('origin'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(r,cmap ='gray')
plt.title('r'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(g,cmap ='gray')
plt.title('g'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(b,cmap ='gray')
plt.title('b'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(r_cut)
plt.title('r_cut'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(image_cut)
plt.title('image_cut'), plt.xticks([]), plt.yticks([])
plt.show()

