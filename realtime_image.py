import numpy as np
import cv2
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

shot_idx = 0
brilho = 1
contraste = 0
detect_face_bool = False
face_blur = 0
face_blur = 0
alpha_slider_max = 100
old_face_blur = 0
old_out_blur = 0
auto_correction = 0
cinzento = 0

def adjust_brightness(frame2, brightness_factor=0, contrast_factor=0):
    brightness_factor /= 100
    contrast_factor -= 127

    if brightness_factor != 0:
        if brightness_factor > 0:
            shadow = brightness_factor
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness_factor
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(frame2, alpha_b, frame2, 0, gamma_b)
    else:
        buf = frame2.copy()

    if contrast_factor != 0:
        f = 131*(contrast_factor + 127)/(127*(131-contrast_factor))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    table = cv2.convertScaleAbs(buf, alpha=brightness_factor, beta=contrast_factor)
    brightness = cv2.hconcat([frame, table])
    cv2.imshow('brightness', brightness)

    
def blur_img(img, factor = 20):
   kW = int(img.shape[1] / factor)
   kH = int(img.shape[0] / factor)
    
   #ensure the shape of the kernel is odd
   if kW % 2 == 0: kW = kW - 1
   if kH % 2 == 0: kH = kH - 1
    
   blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
   return blurred_img

def extract_indexes(length, step_size):
   begin=0
   end=0
   indexes = []
    
   cycles = int(length / step_size)
    
   for i in range(cycles):
       begin = i * step_size
       end = i * step_size+step_size
       #print(i, ". [",begin,", ",end,")")
    
       index = []
       index.append(begin)
       index.append(end)
       indexes.append(index)
    
       if begin >= length: break
       if end > length: end = length
    
   if end < length:
       #print(i+1,". [", end,", ",length,")")
       index = []
       index.append(end)
       index.append(length)
       indexes.append(index)
    
   return indexes


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


def nothing(x):
    pass

cv2.namedWindow('brightness')
cv2.createTrackbar('Luminosidade','brightness',100,200,nothing)  #o Brilho varia entre 0 e 1
cv2.createTrackbar('Contraste','brightness',127,252,nothing)     #o contraste varia enter -127 e + 127
cv2.createTrackbar('Detecting Face', 'brightness',0,1,nothing)
cv2.createTrackbar('Background Blur', 'brightness',0,1,nothing)
cv2.createTrackbar('Face Blur','brightness',0,40,nothing)
cv2.createTrackbar('Global Blur','brightness',0,20,nothing)
cv2.createTrackbar('Cores <-> Cinzento','brightness',0,1,nothing)
cv2.createTrackbar('Auto-Correction','brightness',0,1,nothing)


while(True):
    ret, frame = video.read() # Capture frame by frame of the WebCam
    brightness = cv2.hconcat([frame, frame])
    global_blur = cv2.getTrackbarPos('Global Blur','brightness')
    if global_blur == 0:
        global_blur = 1
    frame2 = cv2.blur(frame, (global_blur, global_blur))
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    brilho = cv2.getTrackbarPos('Luminosidade','brightness')
    contraste = cv2.getTrackbarPos('Contraste','brightness')
    detection_face = cv2.getTrackbarPos('Detecting Face', 'brightness')
    out_blur = cv2.getTrackbarPos('Background Blur','brightness')
    face_blur = cv2.getTrackbarPos('Face Blur','brightness')
    cinzento = cv2.getTrackbarPos('Cores <-> Cinzento','brightness')
    if old_face_blur != face_blur:                              #Altera entre o Blur do background e da cara
        out_blur = 0
        cv2.setTrackbarPos('Background Blur','brightness', 0)
    if old_out_blur != out_blur:
        face_blur = 0
        cv2.setTrackbarPos('Face Blur','brightness', 0)
    old_face_blur = face_blur
    old_out_blur = out_blur
    auto_correction = cv2.getTrackbarPos('Auto-Correction','brightness')

    #Correção Automatica dos Histogramas com o equalizeHist
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    

    faces = faceCascade.detectMultiScale(   #Deteção da Face
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
    
    if detection_face == 1:
        for (x, y, w, h) in faces:      # Draw a rectangle around the faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    if cinzento == 1:  #If TRUE show video in Cinzento
        if auto_correction == 1:
            grayimg = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Coreeção CLAHE (Contrast Limited Adaptive Histogram Equalization) 
            cl1 = clahe.apply(grayimg)
            grayy = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
        else:
            grayy = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        # sp_05 = salt_pepper(grayy, 0.5)
        # cv2.imshow('show', contraharmonic_mean(grayy, sp_05, (3,3), 0.5))
        if out_blur == 1:
            blurred_img = blur_img(grayy, factor = 10)
            for x, y, w, h in faces:
                detected_face = grayy[int(y):int(y+h), int(x):int(x+w)]
                blurred_img[y:y+h, x:x+w] = detected_face
            adjust_brightness(blurred_img, brilho, contraste)
        else:
            if face_blur > 0:
                for x, y, w, h in faces:
                    detected_face = grayy[int(y):int(y+h), int(x):int(x+w)]
                    pixelated_face = detected_face.copy()
                    
                    width = pixelated_face.shape[0]
                    height = pixelated_face.shape[1]

                    step_size = cv2.getTrackbarPos('Face Blur','brightness')

                    for wi in extract_indexes(width, step_size):
                        for hi in extract_indexes(height, step_size):
                            detected_face_area = detected_face[wi[0]:wi[1], hi[0]:hi[1]]
                            if detected_face_area.shape[0] > 0 and detected_face_area.shape[1] > 0:
                                detected_face_area = blur_img(detected_face_area, factor = 0.5)
                                pixelated_face[wi[0]:wi[1], hi[0]:hi[1]] = detected_face_area
                    grayy[y:y+h, x:x+w] = pixelated_face
                adjust_brightness(grayy, brilho, contraste)
            else:
                adjust_brightness(grayy, brilho, contraste)
    else:
        if auto_correction == 1:
            frame2 = img_output
        if out_blur == 1:
            blurred_img = blur_img(frame2, factor = 10)
            for x, y, w, h in faces:
                detected_face = frame2[int(y):int(y+h), int(x):int(x+w)]
                blurred_img[y:y+h, x:x+w] = detected_face
            adjust_brightness(blurred_img, brilho, contraste)
        else:
            if face_blur > 0:

                for x, y, w, h in faces:
                    detected_face = frame2[int(y):int(y+h), int(x):int(x+w)]
                    pixelated_face = detected_face.copy()
                    
                    width = pixelated_face.shape[0]
                    height = pixelated_face.shape[1]

                    step_size = cv2.getTrackbarPos('Face Blur','brightness')

                    for wi in extract_indexes(width, step_size):
                        for hi in extract_indexes(height, step_size):
                            detected_face_area = detected_face[wi[0]:wi[1], hi[0]:hi[1]]
                            if detected_face_area.shape[0] > 0 and detected_face_area.shape[1] > 0:
                                detected_face_area = blur_img(detected_face_area, factor = 0.5)
                                pixelated_face[wi[0]:wi[1], hi[0]:hi[1]] = detected_face_area
                    frame2[y:y+h, x:x+w] = pixelated_face
                adjust_brightness(frame2, brilho, contraste)
            else:
                adjust_brightness(frame2, brilho, contraste)


    ch = 0xFF & cv2.waitKey(1)  # Press 'q' to exit
    if ch == ord('q'):
        break


# Print and save image
    if ch == ord('p'):
        fn = './prints/shot_%03d.bmp' % (shot_idx)
        cv2.imwrite(fn, frame)
        print(fn, 'saved')
        shot_idx += 1

        image = limpaImg('./prints/shot_%03d.bmp' % (shot_idx-1), './prints/shot_%03dFilter.png' % (shot_idx-1))
        print(pytesseract.image_to_string(image))


# Definition of window size
    if ch == ord('1'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    if ch == ord('2'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
 
    if ch == ord('3'):
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


# When everything done, release the capture
video.release()
cv2.destroyAllWindows()