import cv2
import numpy as np 


img = cv2.pyrDown(cv2.imread("lunkuo.jpg",cv2.IMREAD_UNCHANGED))

ret,thresh = cv2.threshold(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)

image ,contours ,hier =cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

    epsilon = 0.01 * cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(img,[approx],0,(0,0,255),3)
    hull= cv2.convexHull(c)
    cv2.drawContours(img,[hull],0,(0,255,255),3)


cv2.drawContours(img,contours,-1,(255,0,0),1)
cv2.imshow("contours",img)    
cv2.waitKey()