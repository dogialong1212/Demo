import numpy as np
import cv2
import glob
from tempfile import TemporaryFile

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)  #Số ô cờ
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('C:/Users/GIA_LONG/Downloads/WPy64-3770/scripts/ball-tracking/ball-tracking/calib/calib[1-8].png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,5),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(21,21),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,5), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#Lưu dữ liệu

np.savez('outfile.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
#_ = outfile.seek(0) # Only needed here to simulate closing & reopening file

#np.savetxt('test.out', ret, mtx, dist, rvecs, tvecs, delimiter=',')
#Xem dữ liệu
#>>> 
#>>> npzfile = np.load(outfile)
#>>> npzfile.files
#['arr_0', 'arr_1']
#>>> npzfile['arr_0']
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

img = cv2.imread('calib7.png')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)
cv2.imshow('img',img)
cv2.imshow('img_cal',dst)
