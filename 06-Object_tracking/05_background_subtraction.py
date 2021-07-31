# 

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2()
# knn = cv2.createBackgroundSubtractorKNN()


while True:
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # apply background subtractor
    mog2_frame = mog2.apply(frame)
    # knn_frame = knn.apply(frame)
    
    # display the frames
    cv2.imshow("Original footage", frame)
    cv2.imshow("MOG2 Frame", mog2_frame)
    # cv2.imshow("KNN frame", knn_frame)
    
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()