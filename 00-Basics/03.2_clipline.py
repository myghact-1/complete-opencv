import cv2
import numpy as np

blank_image = np.zeros(shape=(300, 300, 3), dtype=np.uint8)

# draw a line
cv2.line(blank_image, (0, 0), (300,300), (255,255,255), 3)
# cv2.rectangle(blank_image, (0, 0), (100, 100), (0,0,255), 3)


ret, p1, p2 = cv2.clipLine(imgRect=(0,0,100,100), pt1=(0, 0), pt2=(300, 300))
# print('ret: ', ret)

if ret:
    cv2.line(blank_image, p1, p2, (255,0,0), 3)

cv2.imshow("Image", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()