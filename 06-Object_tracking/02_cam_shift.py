import cv2
import time
import numpy as np

cap = cv2.VideoCapture("images\\car_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print('fps: ', fps)

#* read the first frame 
ret, frame = cap.read()

#* initialize the location
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

#* set up ROI
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([(0., 60., 32.)]), 
                   np.array([(180., 255., 255.)]))

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#* setup the termination criteria
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read the frames...")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    # apply the mean shift
    ret, track_window = cv2.CamShift(dst, track_window, term_criteria)
    
    # draw it on image
    pts = cv2.boxPoints(ret)
    pts =np.int0(pts)
    final_image = cv2.polylines(frame, [pts], True, 255, 2)
    
    # show the frames
    cv2.imshow("Final Image", final_image)
    time.sleep(1/fps)
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()