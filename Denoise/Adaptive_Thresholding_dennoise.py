import cv2
import numpy as np
import pytesseract
import os
from PIL import Image

inp_path = 'img/2.png'
img = cv2.imread(inp_path,0)
blur = cv2.GaussianBlur(img,(3,3),0)

#AdaptiveThreshold Gaussian: the threshold value T(x,y) is a weighted sum
#(cross-correlation with a Gaussian window) of the blockSize×blockSize neighborhood of (x,y) minus C
# with blockSize = 15, C = 20
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,21)
cv2.imshow('Threshold',th3)


#Opening
th3 = cv2.erode(th3,(5,5))
th3 = cv2.dilate(th3,(5,5))



cv2.imshow('Input',img)
cv2.imshow('Output',th3)
# Ghi tạm ảnh xuống ổ cứng để sau đó apply OCR
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, th3)

# Load ảnh và apply nhận dạng bằng Tesseract OCR
text = pytesseract.image_to_string(Image.open(filename))

# Xóa ảnh tạm sau khi nhận dạng
os.remove(filename)

# In dòng chữ nhận dạng được
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()
