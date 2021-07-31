import cv2
import numpy as np


blank_image = np.zeros(shape=(500,500), dtype=np.uint8)

#? Get the size of a text
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 2
thickness = 3

ret, baseline = cv2.getTextSize("OpenCv", font, scale, thickness)
w, h = ret
print('baseline: ', baseline)
print('ret: ', ret)

#? Put the text on image
x1, y1 = 60, 200        #top left point

cv2.putText(blank_image, "OpenCv", (x1, y1), font, scale, (255,255,255), thickness)


#? Draw rectangle outside to the text

cv2.rectangle(blank_image, (x1, y1-h-5), (x1+w, y1+baseline), (255,0,0), 2)


cv2.imshow("Text Image", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()