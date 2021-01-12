import numpy as np
import cv2
from PIL import Image
import pytesseract


video = cv2.VideoCapture(0)

shot_idx = 0
brilho = 1
gray_bool = False


def adjust_brightness(frame2, brightness_factor):
    table = cv2.convertScaleAbs(frame2, alpha=brightness_factor, beta=0)
    brightness = cv2.hconcat([frame, table])
    cv2.imshow('brightness', brightness)

def salt_pepper(grayy, prob):
      # Extract image dimensions
      row, col, c = grayy.shape

      # Declare salt & pepper noise ratio
      s_vs_p = 0.5
      output = np.copy(grayy)

      # Apply salt noise on each pixel individually
      num_salt = np.ceil(prob * grayy.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in grayy.shape]
      output[coords] = 1

      # Apply pepper noise on each pixel individually
      num_pepper = np.ceil(prob * grayy.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in grayy.shape]
      output[coords] = 0
      cv2.imshow('output', output)

      return output

def limpaImg(img1, novaImg):
    img = Image.open(img1)
    img = img.point(lambda x: 0 if x<100 else 255)
    img.save(novaImg)    
    return img

# def contraharmonic_mean(grayy, img, size, Q):
#     num = np.power(grayy, Q + 1)
#     denom = np.power(grayy, Q)
#     kernel = np.full(size, 1.0)
#     result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
#     return result

while(True):
    ret, frame = video.read() # Capture frame by frame of the WebCam
    brightness = cv2.hconcat([frame, frame])
    frame2 = cv2.blur(frame, (2, 2))
    

    if gray_bool == True:  #If TRUE show video in GRAY
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        grayy = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        # sp_05 = salt_pepper(grayy, 0.5)
        # cv2.imshow('show', contraharmonic_mean(grayy, sp_05, (3,3), 0.5))
        adjust_brightness(grayy, brilho)
    else:
        adjust_brightness(frame2, brilho)
    

    ch = 0xFF & cv2.waitKey(1)  # Press 'q' to exit
    if ch == ord('q'):
        break


### Key Press Actions

# Light
    if ch == ord('+'):
        brilho = brilho * 1.1
    if ch == ord('-'):
        brilho = brilho * 0.9


# Colors
    if ch == ord('g'):  #Image to GRAY
        gray_bool = True

    if ch == ord('c'):  ##Image with COLORS
        gray_bool = False


# Print and save image
    if ch == ord('p'):
        fn = './prints/shot_%03d.bmp' % (shot_idx)
        cv2.imwrite(fn, frame)
        print(fn, 'saved')
        shot_idx += 1

        image = limpaImg('./prints/shot_%03d.bmp' % (shot_idx-1), './prints/shot_%03dFilter.png' % (shot_idx-1))
        print(pytesseract.image_to_string(image))


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