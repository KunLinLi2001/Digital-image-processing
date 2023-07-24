import cv2
'''
图片的读取：img = cv2.imread(filename,flags)
——filename：读入图片的路径（保证路径中没有中文）
——flags：读入图片的标志（可以省略）
cv2.IMREAD_COLOR：默认参数，读入彩色图片，忽略alpha通道
cv2.IMREAD_GRAYSCALE：读入灰度图片
cv2.IMREAD_UNCHANGED：读入完整图片，包括alpha通道
'''
filename = "2.bmp"
# img = cv2.imread(filename,0)
# '''
# cv2.imshow("catcat",img)
# ——第一个参数是显示图像的窗口的名字
# ——第二个是参数要显示的图像（imread读入的图像）
# cv2.waitKey(0)
# ——等待键盘输入，单位为毫秒
# ——参数为0表示无限等待
# '''
# cv2.imshow("catcat",img)
# cv2.waitKey(0)
# '''
# cv2.imwrite(file，img，num) 
# --第一个参数是要保存的文件名
# --第二个参数是要保存的图像
# --可选的第三个参数，它针对特定的格式：
# 对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95;
# 对于png ,第三个参数表示的是压缩级别，默认为3。
# 注意: • cv2.IMWRITE_JPEG_QUALITY类型为 long ,必须转换成 int 
#      • cv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小
# '''
# cv2.imwrite("dogegg.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),95])
