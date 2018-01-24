import cv2
import numpy as np 


img = cv2.pyrDown(cv2.imread("lunkuo.png",cv2.IMREAD_UNCHANGED))

ret,thresh = cv2.threshold(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)

image ,contours ,hier =cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    #计算简单的边界框
    x,y,w,h=cv2.boundingRect(c)
    #划出矩形
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #通过minAreaRect找出包围目标的最小矩形区域
    #Opencv没有函数能直接从轮廓信息中算出最小矩形的定点坐标，
    # 所有需要计算出最小矩形区域，然后计算这个矩形的顶点
    #有雨计算出来的定点坐标是浮点型，但像素坐标是整形，所以需要一个转换
    rect = cv2.minAreaRect(c)
    box =cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img,[box],0,(0,0,255),3)
    #计算圆心点和半径通过minEnclosingCircle获取边界轮廓的最小闭圆
    (x,y),radius = cv2.minEnclosingCircle(c)

    center =(int(x),int(y))

    radius = int(radius)

    img = cv2.circle(img,center,radius,(0,255,0),2)

cv2.drawContours(img,contours,-1,(255,0,0),1)
cv2.imshow("contours",img)    
cv2.waitKey()