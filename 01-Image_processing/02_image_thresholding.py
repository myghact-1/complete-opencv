
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

# --------------------------- Thresholding Methods --------------------------- #
#* 1. Global thresholding
#* 2. Adaptive thresholding
#* 3. Otsu's thresholding
# ---------------------------------------------------------------------------- #


import cv2

image = cv2.imread("images\\threshold.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sudoku_gray = cv2.imread("images\\sudoku.jpg", 0)
otsu_gray = cv2.imread("images\\otsu.jpg", 0)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Global / Simple Thresholding

_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)
_, thresh_tozero_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)

show_image("Original Grayscale image", gray)
show_image("Binary Threshold", thresh_binary)
show_image("Binary inverse", thresh_binary_inv)
show_image("Thresh Truncated", thresh_trunc)
show_image("Thresh tozero", thresh_tozero)
show_image("Thresh tozero inverse", thresh_tozero_inv, destroy=True)

# ---------------------------------------------------------------------------- #
#? Adaptive Thresholding

_, simple_thresh = cv2.threshold(sudoku_gray, 127, 255, cv2.THRESH_BINARY)

adaptive_thresh_mean = cv2.adaptiveThreshold(sudoku_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

adaptive_thresh_gaussian = cv2.adaptiveThreshold(sudoku_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

show_image("Original Sudoku Image", sudoku_gray)
show_image("Simple Threshold", simple_thresh)
show_image("Thresh mean adaptive", adaptive_thresh_mean)
show_image("thresh gaussian adaptive", adaptive_thresh_gaussian, destroy=True)


# ---------------------------------------------------------------------------- #
#? Otsu's Thresholding

_, th1 = cv2.threshold(otsu_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# apply blurring
blurred = cv2.GaussianBlur(otsu_gray, (5,5), 0)
_, th2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

show_image("Original Image", otsu_gray)
show_image("Otsu's image", th1)
show_image("Otsu's image after blur", th2, destroy=True)

# ---------------------------------------------------------------------------- #