
import cv2
import numpy as np

im = cv2.imread('shot_001.bmp')
img = cv2.imread('shot_001.bmp',0)

# Convert grayscale image to 3-channel image,so that they can be stacked together    
imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
both = np.hstack((im,imgc))

cv2.imshow('imgc',both)
cv2.waitKey(0)
cv2.destroyAllWindows()
