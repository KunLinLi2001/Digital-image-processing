# 11.18 OTSU 阈值处理算法的实现
img = cv2.imread("../images/Fig1039a.tif", flags=0)

histCV = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
scale = range(256)  # 灰度级 [0,255]
totalPixels = img.shape[0] * img.shape[1]  # 像素总数
totalGray = np.dot(histCV[:, 0], scale)  # 内积, 总和灰度值
mG = totalGray / totalPixels  # 平均灰度

icv = np.zeros(256)
numFt, sumFt = 0, 0
for t in range(256):  # 遍历灰度值
    numFt += histCV[t, 0]  # F(t) 像素数量
    sumFt += histCV[t, 0] * t  # F(t) 灰度值总和
    pF = numFt / totalPixels  # F(t) 像素数占比
    mF = (sumFt / numFt) if numFt > 0 else 0  # F(t) 平均灰度
    numBt = totalPixels - numFt  # B(t) 像素数量
    sumBt = totalGray - sumFt  # B(t) 灰度值总和
    pB = numBt / totalPixels  # B(t) 像素数占比
    mB = (sumBt / numBt) if numBt > 0 else 0  # B(t) 平均灰度
    icv[t] = pF * (mF - mG) ** 2 + pB * (mB - mG) ** 2  # 灰度 t 的类间方差
maxIcv = max(icv)  # ICV 的最大值
maxIndex = np.argmax(icv)  # 最大值的索引

# 阈值处理
ret, imgBin = cv2.threshold(img, maxIndex, 255, cv2.THRESH_BINARY)  # 以 maxIndex 作为最优阈值
ret, imgOtsu = cv2.threshold(img, mG, 255, cv2.THRESH_OTSU)  # 阈值分割, OTSU
print("t(maxICV)={}, retOtsu={}".format(maxIndex, round(ret)))

plt.figure(figsize=(7, 7))
plt.subplot(221), plt.axis('off'), plt.title("Origin"), plt.imshow(img, 'gray')
plt.subplot(222, yticks=[]), plt.title("Gray Hist")  # 直方图
plt.plot(scale, histCV[:, 0] / max(histCV))  # 灰度直方图
plt.plot(scale, icv / maxIcv)  # 类间方差图
plt.subplot(223), plt.title("global binary(T={})".format(maxIndex)), plt.axis('off')
plt.imshow(imgBin, 'gray')
plt.subplot(224), plt.title("OTSU binary(T={})".format(round(ret))), plt.axis('off')
plt.imshow(imgOtsu, 'gray')
plt.tight_layout()
plt.show()
