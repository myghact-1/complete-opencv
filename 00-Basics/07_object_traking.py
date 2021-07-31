import cv2
import numpy as np


# ------------------------- Steps in object tracking ------------------------- #
#* 1. Read the frame(s) / image(s)
#* 2. change color channels to HSV
#* 3. define hsv color bounds for object
#* 4. create a mask of hsv by using 'inRange' function
#* 5. use bitwise and operation for final result
# ---------------------------------------------------------------------------- #

#* https://nalinc.github.io/blog/2018/skin-detection-python-opencv/



# --------------------------------- Using HSV -------------------------------- #
#? Object tracking (Skin detection)

def object_tracking_hsv(address, lower, upper):
    """
    HSV object tracking of any object

    Args:
        address (int or str): path of the video or camera, int '0' for built-in webcam
        lower (1-D arrar): lower bound color range of the object
        upper (1-D arrar): upper bound color range of the object
    """    
    cap = cv2.VideoCapture(address)

    if not cap.isOpened():
        print("Didn't find camera!!!")
        exit()
        

    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print("Can't read the frames...")
            break
        
        # RBG ---> HSV color change
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # skin color bounds for HSV
        # lower_bound = np.array([0, 58, 30], dtype = "uint8")
        # upper_bound = np.array([33, 255, 255], dtype = "uint8")
        
        # Mask for skin color
        mask = cv2.inRange(hsv, lower, upper)
        
        # bitwise operation for final result
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Show the frames
        cv2.imshow("Frame", frame)
        cv2.imshow("Skin", res)
        
        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

lower_bound = np.array([0, 58, 30], dtype = "uint8")
upper_bound = np.array([33, 255, 255], dtype = "uint8")

# object_tracking_hsv(address=0, lower=lower_bound, upper=upper_bound)



# -------------------------------- Using YCrCb -------------------------------- #
#? Skin Detection

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Didn't find camera!!!")
    exit()
    

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Can't read the frames...")
        break
    
    # RBG ---> HSV color change
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # skin color bounds for HSV
    lower_YCrCb = np.array([0,133,77],np.uint8)
    upper_YCrCb = np.array([235,173,127],np.uint8)
    
    # Mask for skin color
    mask = cv2.inRange(ycrcb, lower_YCrCb, upper_YCrCb)
    
    # bitwise operation for final result
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Show the frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Skin", res)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()