from os import terminal_size
from typing import Text
import cv2
import numpy as np



#? Drawing a circle by mouse click

def draw_circle(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 30, (255, 0, 70), -1)

# create blank image and window
image = np.zeros(shape=(512,512,3), dtype=np.uint8)
cv2.namedWindow("Circle")
cv2.setMouseCallback("Circle", draw_circle)

while True:
    cv2.imshow("Circle", image)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------- #


#? Drawing Rectangles / Squares

drawing = False
x1, y1 = -1, -1

def draw_shape(event, x, y, flags, param):
    global x1, y1, drawing
    
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        
    if event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(image, (x1, y1), (x, y), (255,255,255), -1)
            
    if event==cv2.EVENT_LBUTTONUP:
        drawing = False
        
# create blank image and window
image = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('rectangle')
cv2.setMouseCallback('rectangle', draw_shape)

while True:
    cv2.imshow("rectangle", image)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()