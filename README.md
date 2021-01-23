# Real_Time_Image_Transformation

This project has as main objective the modification of images in real time.
The developed application allows changing, in real time, video images acquired by a camera.

## For the user to interact with the application, it is necessary to **install** two libraries:
> sudo pip install pillow
> sudo apt install tesseract-ocr

### If the user is unable to install the second library, he must install: 
> sudo add-apt-repository ppa: alex-p / tesseract-ocr-devel” and then “sudo apt update.

### The *facecascade API* is used for the detection of faces by the camera, a *Tesserac library* for recognizing the text of the contrasted image we have produced and a *PIL library* for reading .bmp images from the Print folder.
