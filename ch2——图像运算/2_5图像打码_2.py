import cv2
img = cv2.imread('pictures/dundun.jpg',0)
height = img.shape[0] # 确定高度
width = img.shape[1] # 确定宽度
print(height,width)

# 双重for循环遍历所有像素点
for i in range(0, height):
    for j in range(0, width):
        if i % 5 == 0 and j % 5 == 0:  # i=1915 j=860时有bug
            for m in range(0, 5):
                for n in range(0, 5):
                    if i + m >= height or j + n >= width:
                        break;
                    level = img[i, j]
                    img[i + m, j + n] = level

cv2.imshow('mosaic',img)
cv2.waitKey(0)