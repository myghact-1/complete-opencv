import cv2


#? Image reading "BGR" Channels
image = cv2.imread("images\\2.jpg")


#? Image Showing in a window
cv2.imshow("Image", image)
cv2.waitKey(0) # wait for infinite time for next operation
# Press any key to proceed next operation
cv2.destroyAllWindows()


#? Image Writing / Saving
cv2.imwrite("images\\2.1.jpg", image)