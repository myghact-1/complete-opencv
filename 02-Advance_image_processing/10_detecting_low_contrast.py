
import cv2
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import imutils


image_paths = list(list_images("images\\examples"))


# loop over the image paths

for i, path in enumerate(image_paths):
    # load the input image from disk, resize it and convert it to grayscale
    image = cv2.imread(path)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # blur the image and perform edge detection
    blurred = cv2.bilateralFilter(gray, 5, 50, 70)
    edged = cv2.Canny(blurred, 30, 150)
    
    # initialize the text 
    text = "Low contrast: No"
    color = (0,255,0)
    
    # check image contrast
    if is_low_contrast(edged.copy(), fraction_threshold=0.1):
        text = "Low contrast: Yes"
        color = (0, 0, 255)
        
    else:
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        # select the max contour and draw it
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, [cnt], -1, (0,255,0), 2)
        
    # Draw the test on the output image
    cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # display the output image
    cv2.imshow("original Image", image)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()