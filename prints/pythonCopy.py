import cv2
import numpy as np
from PIL import Image
import pytesseract
import cv2
import glob
import requests
import matplotlib.pyplot as plt


i = 0

image = cv2.imread("ImgLimpa.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

t_y,b_y,l_x  = 310, 380, 490
cropped_image = image[t_y:b_y,l_x:, :]

# converting to buffer
_, encoded_image = cv2.imencode('.png', cropped_image)
buffer = encoded_image.tobytes()

# request params
KEY = '6d4c118792804b1097eaaa66b5e3dd92'
visionBaseUrl = 'https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/'
ocrUrl = visionBaseUrl + "ocr"

headers = {'Ocp-Apim-Subscription-Key': KEY,
        'Content-Type': 'application/octet-stream'}
params  = { 'detectOrientation': 'true'}


try:
    response = requests.post(ocrUrl, headers=headers, params=params, data=buffer)
    response.raise_for_status()

    analysis = response.json()
    print(analysis)
except Exception as e:
    raise Exception(e)
