
#? Lucas-Kanade Optical Flow

import cv2
import time
import numpy as np

# params for Shitomasi corner detection
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

# parameters for lukas kanade optical flow
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# create some random colors
colors = np.random.randint(0, 255, (100,3))

cap = cv2.VideoCapture("images\\car_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

# Take the first frame and find corners in it
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)

# create a mask image for drawing
mask = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frame, gray, p0, None, **lk_params)
        
        # select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), colors[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a),int(b)), 5, colors[i].tolist(), -1)
            
        img = cv2.add(frame, mask)
    
        cv2.imshow("frame", img)
        time.sleep(1/fps)
        
        # Now update the previous frame and previous points
        old_gray = gray.copy()
        p0 = good_new.reshape((-1,1,2))
        
    else:
        break
        
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()