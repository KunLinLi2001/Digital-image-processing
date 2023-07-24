import cv2
n = int(input("请输入要压缩的级别："))
img = cv2.imread('pictures/ying.jpg')
height = img.shape[0] # 确定高度
width = img.shape[1] # 确定宽度
print(height,width)

def mosaic(size):
    global img
    # 双重for循环遍历所有像素点
    for i in range(0, height):
        for j in range(0, width):
            if i % size == 0 and j % size == 0:
                for m in range(0, size):
                    for n in range(0, size):
                        if i + m >= height or j + n >= width:
                            break;
                        (b, g, r) = img[i, j]
                        img[i + m, j + n] = (b, g, r)
mosaic(n)
cv2.imshow('mosaic',img)
cv2.waitKey(0)