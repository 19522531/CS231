import cv2
import numpy as np


def tack( p_persion ,p_bg):
    image = cv2.imread(p_bg,1)
    ##image = cv2.resize(image, (300, 300))

    frame = cv2.imread(p_persion,1)
    ##frame = cv2.resize(frame, (300, 300))
    
    image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x = np.array([0,137,123])
    y = np.array([107,255,255])

    Mask = cv2.inRange(hsv,x,y)
    res = cv2.bitwise_and(frame, frame, mask = Mask)
    f = frame - res
    not_mask = cv2.bitwise_not(Mask)
    img_gb2 = cv2.bitwise_and(image,image,mask= Mask)
    imt = f + img_gb2
    
    return imt



img = tack("static/mesi.jpg", "static/iStock-517188688.jpg" )
cv2.imshow("img", img)

cv2.waitKey(0)