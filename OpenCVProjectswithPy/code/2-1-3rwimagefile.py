import numpy
import cv2
import os

# 制作一个120,000个随机字节的数组
randomByteArray = bytearray(os.urandom(120000))

flatNumpyArray = numpy.array(randomByteArray)

# 转换数组以制作400x300灰度图像。
grayImage = flatNumpyArray.reshape(300,400)
cv2.imwrite("randomGray.png",grayImage)

# 转换数组以制作100x300的bgr彩色图像。
bgrImage = flatNumpyArray.reshape(100,400,3)
cv2.imwrite("randomColor.png",bgrImage)