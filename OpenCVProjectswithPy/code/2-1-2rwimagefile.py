import cv2

grayImage = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('2Gray.png', grayImage)