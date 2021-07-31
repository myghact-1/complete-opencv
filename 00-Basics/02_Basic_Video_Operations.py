import cv2

#? Video Reading from webcam

def video_preview(address):
    """
    This function preview a video in a window of any webcam.

    Args:
        address (int or str): integer value like '0' for built-in webcam 
        or string path of the video file and/or ip webcam
    """        
        
    cap = cv2.VideoCapture(address)

    if not cap.isOpened():
        print("Didn't found a camera!!!")
        exit()
        
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frames...")
            break
        
        cv2.imshow(f"My Video {height}x{width} - FPS: {fps}", frame)
        
        # Press 'q' to close the window
        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

# video_preview(address=0)


#? Video Reading from IP WEBCAM

ip_address = "https://100.83.232.31:8080/video"

# video_preview(address=ip_address)


#? Video Reading from a file

path = "images\\PexelsVideos1.mp4"

# video_preview(address=path)


#? Video Writing / Saving

cap = cv2.VideoCapture(0)
forcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("images\\output.avi", forcc, 20.0, (640,  480))

if not cap.isOpened():
    print("Didn't find a camera!!!")
    exit()
    
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frames...")
        break
    
    cv2.imshow("My Video", frame)
    out.write(frame)
    
    # Press 'q' to close the window
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()