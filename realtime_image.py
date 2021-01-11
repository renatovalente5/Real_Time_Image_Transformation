import numpy as np
import numpy as _is_numpy_image
import cv2


video = cv2.VideoCapture(0)

shot_idx = 0
brilho = 1

def adjust_brightness(img, brightness_factor):
    table = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    brightness = cv2.hconcat([frame, table])
    cv2.imshow('brightness', brightness)


while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = cv2.hconcat([frame, frame])
    cv2.namedWindow('brightness')
    adjust_brightness(frame, brilho)

    ch = 0xFF & cv2.waitKey(1)
    if ch == ord('q'):
        break


### Key Press Actions

# Light
    if ch == ord('+'):
        brilho = brilho * 1.1
    if ch == ord('-'):
        brilho = brilho * 0.9

        
# Print and save image
    if ch == ord('p'):
        fn = './prints/shot_%03d.bmp' % (shot_idx)
        cv2.imwrite(fn, brightness)
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