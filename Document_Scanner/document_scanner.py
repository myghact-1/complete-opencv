import numpy as np 
import time
import cv2
from utils import get_selected_roi

cap = cv2.VideoCapture("http://10.16.135.216:8080/video")

if cap.isOpened() == False:
    print("Error in opening video stream or file")
    exit()

# The function attached to the trackbar.
def func(pos):
    pass

cv2.namedWindow("Image")
cv2.createTrackbar('Threshold1', "Image", 0, 255, func)
cv2.createTrackbar('Threshold2', "Image", 0, 255, func)


while True:
    
    # reading camera frames
    ret, frame = cap.read()
    # if frame reading successful
    if ret:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Gaussian blur for canny edges detection
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        
        th1 = cv2.getTrackbarPos("Threshold1","Image")
        th2 = cv2.getTrackbarPos("Threshold2","Image")
        
        # detecting edges
        edges = cv2.Canny(blur, th1, th2)
        
        # finding all contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sorting contours by area in descending order
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = contours[0] # selecting contour with maximum area
    
    # show edges and frame
    cv2.imshow("Image", edges)
    frame_copy = frame.copy()
    text1 = "Press 'c' for outer edges selection"
    text2 = "and press 's' for saving"
    text3 = "Press 't' for BW image and saving"
    frame_copy = cv2.putText(frame_copy, text1, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))
    frame_copy = cv2.putText(frame_copy, text2, (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))
    frame_copy = cv2.putText(frame_copy, text3, (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))
    cv2.imshow("Contour", frame_copy)
    
    k = cv2.waitKey(1)
    if k==ord('q'): # press 'q' to exit program
        break
    elif k==ord('c'): # for getting maximum area contour's selection
        # getting corner points
        epsilon = 0.01*cv2.arcLength(cnt, True)
        points = cv2.approxPolyDP(cnt, epsilon, True)
        if len(points)==4:
            dst = get_selected_roi(frame, points)
            cv2.imshow("Final", dst)
    elif k==ord('s'): # saving selected part
        if dst.max():
            name = str(int(time.time()))
            cv2.imwrite(f"Projects\\saved\\{name}.jpg", dst)
        else:
            pass
        
    elif k==ord('t'): # selection thresholding
        if dst.max():
            name = str(int(time.time()))
            dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow("Threshold", thresh)
            cv2.imwrite(f"Projects\\saved\\{name}_BW.jpg", thresh)
        else:
            pass
            
    
cap.release()
cv2.destroyAllWindows()