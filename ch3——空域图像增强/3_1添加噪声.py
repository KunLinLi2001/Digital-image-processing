import cv2
import skimage # 处理以numpy矩阵为载体的图像
import numpy as np

# 1.高斯噪声（自带库函数）
# img1 = skimage.io.imread("pictures/ying.jpg")
img1 = cv2.imread("wife.jpg")
noisy1 = skimage.util.random_noise(img1,mode='gaussian',var=0.01)
# util:utility的缩写，代表一些使用的应用集  random_noise随机噪声
# mode：选择模式  var：方差 'gaussian'为高斯噪声，服从标准方差为0.01的高斯分布（正态分布）
cv2.imshow("gaussian1",noisy1)

# 2.高斯噪声（手写）
def gaussian(img,var=0.01):
  img = np.array(img/255, dtype=float)
  noise = np.random.normal(0,var ** 0.5,img.shape)
  out = img + noise
  out = np.clip(out, 0, 1.0)
  out = np.uint8(out*255)
  return out
img2 = cv2.imread("wife.jpg")
noisy2 = gaussian(img2,0.01)
cv2.imshow("gaussian2",noisy2)

# 3.椒盐噪声（自带库函数）
img3 = cv2.imread("wife.jpg")
noisy3 = skimage.util.random_noise(img3,mode='s&p',salt_vs_pepper=0.5)
cv2.imshow("pepper1",noisy3)

cv2.waitKey(0)