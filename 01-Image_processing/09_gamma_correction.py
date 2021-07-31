import cv2
from skimage import exposure

image = cv2.imread("images\\desert.jpg")

gamma_correction = exposure.adjust_gamma(image, gamma=0.5)

cv2.imshow("Original Image", image)
cv2.imshow("Adjusted image", gamma_correction)

cv2.waitKey(0)
cv2.destroyAllWindows()