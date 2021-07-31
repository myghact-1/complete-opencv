
# https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html


import cv2
import numpy as np
from numpy.core.fromnumeric import shape

# https://www.pexels.com/@mikebirdy
image = cv2.imread("images\\car.jpg")


drawing = False
mode = True

def draw_circles(event, x, y, flags, param):
    global drawing, mode
    
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
    if event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.circle(image_copy, (x,y), 2, (255,255,255), -1) # select foreground
                cv2.circle(mask_copy, (x,y), 2, 255, -1) # select foreground
                
            else:
                cv2.circle(image_copy, (x,y), 2, (0,0,0), -1) # select background
                cv2.circle(mask_copy, (x,y), 2, 0, -1) # select background
                
                
    if event==cv2.EVENT_LBUTTONUP:
        drawing = False

image = cv2.imread("images\\car.jpg")
mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)
mask_copy = mask.copy()
image_copy = image.copy()

cv2.namedWindow("extraction")
cv2.setMouseCallback('extraction', draw_circles)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (15, 100, 485, 305)
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image*mask2[:,:,np.newaxis]

while True:
    cv2.imshow("extraction", image)
    cv2.imshow("Orininal", image_copy)
    cv2.imshow("mask copy", mask_copy)
    
    
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    if k==ord('c'):
        mode = not mode
        
        
        
mask[mask_copy==0] = 0
mask[mask_copy==255] = 1
    
mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, 
                                           fgdModel, 5, 
                                           cv2.GC_INIT_WITH_MASK)
    
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image*mask[:,:,np.newaxis]


cv2.imshow("extraction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()