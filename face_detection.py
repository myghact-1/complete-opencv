import cv2
import warnings
warnings.filterwarnings("ignore")

# ip_address = "https:10.57.130.244:8080/video"
cap  = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera no found!!!")
    exit()
cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')


while True:
    
    ret, frame = cap.read()
        
    if not ret:
        print("Can't streaming video...")
        break
    
    # resize the frame
    frame = cv2.resize(frame, (600, 500))
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the faces and draw them
    coor = cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in coor:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
    # show the frames
    cv2.imshow("Detected Faces", frame)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()