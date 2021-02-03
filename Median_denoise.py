import cv2
from scipy import signal,ndimage
import numpy as np
import pytesseract
import os
from PIL import Image

#Link hinh input
#inp_path = 'input/train/2.png'

inp_path = 'input/test3.png'
img = cv2.imread(inp_path)
img = cv2.resize(img,(540,258))

median = cv2.medianBlur(img,11)

cv2.imshow('Median',median)
img_num = np.asarray(img) / 255.0
mask = np.asarray(img) / 255.0
median_num = np.asarray(median) / 255.0
back = np.average(median_num);


for i in range(img_num.shape[0]):
    for j in range(img_num.shape[1]):
        for  k in range(3):
            if (img_num[i, j, k] > median_num[i, j, k] -0.1) :
                mask[i,j,k] = 0


out = np.where(mask,img_num,1.0)
out = np.asarray(out*255.0, dtype=np.uint8)
mask = np.asarray(mask*255.0, dtype=np.uint8)
img_num = np.asarray(img_num*255.0, dtype=np.uint8)

cv2.imshow('Foreground Mask',mask)
cv2.imshow('Input',img_num)
cv2.imshow('Output',out)

# Ghi tạm ảnh xuống ổ cứng để sau đó apply OCR
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, out)

# Load ảnh và apply nhận dạng bằng Tesseract OCR
text = pytesseract.image_to_string(Image.open(filename))

# Xóa ảnh tạm sau khi nhận dạng
os.remove(filename)

# In dòng chữ nhận dạng được
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()