# 彩色图像分割：HSI用饱和度做模板，RGB设置不同阈值
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# RGB模式设置不同阈值彩色图像分割
colorseg(method, img, 0.2, parameters)
img = imread('frog.jpg')
mask = roipoly(f)
red = immultiply(mask, f(:,:, 1))
green = immultiply(mask, f(:,:, 2))
blue = immultiply(mask, f(:,:, 3))
res = cat(3, red, green, blue)
imshow(res)
[M, N, K] = size(res)
I = reshape(res, M * N, 3)
idx = find(mask)
I = double(I(idx, 1:3))
[C, m] = covmatrix(I)
d = diag(C)
sd = sqrt(d)
E25 = colorseg('euclidean', f, 25, m)
E50 = colorseg('euclidean', f, 50, m)
E75 = colorseg('euclidean', f, 75, m)
E100 = colorseg('euclidean', f, 100, m)
subplot(2, 3, 1), imshow(f), title('原图')
subplot(2, 3, 2), imshow(g), title('选择的区域')
f = tofloat(f)
subplot(2, 3, 3), imshow(cat(3, f(:,:, 1).*E25, f(:,:, 2).*E25, f(:,:, 3).*E25)), title('T为25')
subplot(2, 3, 4), imshow(cat(3, f(:,:, 1).*E50, f(:,:, 2).*E50, f(:,:, 3).*E50)), title('T为50')
subplot(2, 3, 5), imshow(cat(3, f(:,:, 1).*E75, f(:,:, 2).*E75, f(:,:, 3).*E75)), title('T为75')
subplot(2, 3, 6), imshow(cat(3, f(:,:, 1).*E100, f(:,:, 2).*E100, f(:,:, 3).*E100)), title('T为100')

cv.waitKey(0)
cv.destroyAllWindows()
