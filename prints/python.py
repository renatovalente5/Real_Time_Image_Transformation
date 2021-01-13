import cv2
import numpy as np
from PIL import Image
import pytesseract

#img = cv2.imread("ImgLimpa.png")

def limpaImg(img1, novaImg):
    img = Image.open(img1)    #Threshold para a imagem, e salva-la
    img = img.point(lambda x: 0 if x<100 else 255)
    img.save(novaImg)    
    return img

image = limpaImg('shot_000.bmp', 'ImgLimpa2.png')
print(pytesseract.image_to_string(image))

# def shadow_remove(img):
#     rgb_planes = cv2.split(img)
#     result_norm_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#         result_norm_planes.append(norm_img)
#     shadowremov = cv2.merge(result_norm_planes)
#     return shadowremov#Shadow removal
# shad = shadow_remove(img)
# cv2.imwrite('shot_0002.bmp', shad)
