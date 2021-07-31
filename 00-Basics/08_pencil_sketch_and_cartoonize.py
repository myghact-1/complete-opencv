import cv2

image = cv2.imread("images\\cow.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Laplacian
laplacian = cv2.Laplacian(gray, -1, ksize=3)
show_image("Laplacian", laplacian, destroy=True)


#? Pencil Sketch
gray_sketch, color_sketch = cv2.pencilSketch(image, sigma_s=10, sigma_r=0.1, shade_factor=0.1)
show_image("Gray Sketch", gray_sketch)
show_image("color_sketch", color_sketch, destroy=True)


#? Stylization
cartoonize = cv2.stylization(image, sigma_s=10, sigma_r=0.1)
show_image("Cartoonized image", cartoonize, destroy=False)