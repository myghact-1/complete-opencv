import cv2

image = cv2.imread("images\\opencv3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5,5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

cv2.imshow("Thresholded image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()