import cv2
import numpy as np

blank_image = np.zeros(shape=(512, 512, 3))

#?  Line drawing
image = cv2.line(blank_image, (0, 0), (512, 512), (255,0,0), 5, cv2.LINE_AA)

#? Draw arrowed line
image = cv2.arrowedLine(image, (0, 512), (256, 256), (255, 255, 0), 5)

#? Drawing rectangle / Square
image = cv2.rectangle(image, (50, 20), (100, 100), (255, 255, 255), 4) 
# Draw Filled Rectangle
image = cv2.rectangle(image, (120, 20), (170, 100), (255, 255, 255), -1)

#? Drawing Circle
image = cv2.circle(image, (300, 150), 50, (0,255,0), -4) # 4--> -1 for filled circle

#? Drawing Ellipse
image = cv2.ellipse(image, (400,400), (100, 50), 0, 0, 360, (255, 30, 70), 4)

#? Drawing Polynomials
points = np.array([[100,50],[200,300],[180,150],[350,200]], np.int32)
points = points.reshape((-1, 1, 2))
image = cv2.polylines(image, pts=[points], isClosed=True, color=(0,0,255), thickness=5)


#? Writing Text on Image
image = cv2.putText(image, "Great!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 90, 10), 5)


cv2.imshow("Image Drawing", image)
cv2.waitKey(0)
cv2.destroyAllWindows()