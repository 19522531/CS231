import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time
from datetime import datetime

# segmentor = SelfiSegmentation()

# def remove_background( path):
#     img  = cv2.imread(path)
#     imgOut = segmentor.removeBG(img, (255,255,255), threshold=0.3)
#     return imgOut    
    
# img = remove_background("static/mesi.jpg")
# cv2.imwrite("static/ok.jpg", img)

time_ = datetime.now()
print(str(time_))