import numpy as np
import cv2

shot_idx = 0
video = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #cv2.imshow('frame', gray)

    ch = 0xFF & cv2.waitKey(1)
    if ch == ord('q'):
        break


## Key Press Actions
 
    if ch == ord('p'):
    
        fn = './prints/shot_%03d.bmp' % (shot_idx)
        cv2.imwrite(fn, frame)
        print(fn, 'saved')
        shot_idx += 1
 

# Definition of window size
    if ch == ord('2'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    if ch == ord('3'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
 
    if ch == ord('4'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()