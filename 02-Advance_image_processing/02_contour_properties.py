# https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html

import cv2
import numpy as np

image  = cv2.imread("images\\one_shape.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Find countour and draw

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cnt = contours[0]
result = cv2.drawContours(image.copy(), [cnt], 0, (0,255,0), 2)

show_image("Image with contours", result, destroy=True)


# ---------------------------------------------------------------------------- #
#? Aspect Ratio 

x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print('aspect_ratio: ', aspect_ratio)


#? Extent

area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area)/rect_area
print('extent: ', extent)


#? Solidity

area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print('solidity: ', solidity)


#? Equivalent Diameter

area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
print('equi_diameter: ', equi_diameter)


#? Orientation

(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
print('angle: ', angle)


#? Mask and Pixel Points

mask = np.zeros(image.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)


#? Maximum Value, Minimum Value and their locations

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask = mask)

#? Mean Color or Mean Intensity
# mean_val = cv2.mean(image, mask = mask)

#? Extreme Points

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

cv2.circle(image, leftmost, 5, (255,0,0),-1)
cv2.circle(image, rightmost, 5, (0,255,0),-1)
cv2.circle(image, topmost, 5, (0,0,255),-1)
cv2.circle(image, bottommost, 5, (255,100,0),-1)

show_image("Extreme points", image, destroy=True)

# ---------------------------------------------------------------------------- #