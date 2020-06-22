# python Object_detection_V3.py --image sample_1.png --width 2.200
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)
    
            ###############Image Processing Sample##################
def find_RefObj(image,knownWidth):
   box = find_marker(image)
   box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
   box = np.array(box, dtype="int")
   # order the points in the contour such that they appear
   # in top-left, top-right, bottom-right, and bottom-left
   box = perspective.order_points(box)
   
   # unpack the ordered bounding box, then compute the
   # midpoint between the top-left and top-right points,
   # followed by the midpoint between the top-right and
   # bottom-right
   cX = np.average(box[:, 0])
   cY = np.average(box[:, 1])
   (tl, tr, br, bl) = box
   (tlblX, tlblY) = midpoint(tl, bl)
   (trbrX, trbrY) = midpoint(tr, br)

   # compute the Euclidean distance between the midpoints,
   # then construct the reference object
   D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
   refObj =  D / knownWidth
               
	
   return refObj

###########Main Program##############
image = cv2.imread(args["image"])
PixperMet = find_RefObj(image,args["width"])
cap = cv2.VideoCapture(1)
while(True):
        ret, image = cap.read()
        marker = find_marker(image)
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        box = perspective.order_points(box)
        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])
        # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # compute the size of the object
        dimA = dA / PixperMet
        dimB = dB / PixperMet
        #find theta ( -45<theta<45)
        theta = marker[-1]
        if theta < -45:
            theta = -(90 + theta)
        else:
            theta = -theta
	# draw the object sizes on the image
        cv2.circle(orig, (int(cX), int(cY)), 4, (0, 255, 0), -1)
        cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
        cv2.putText(orig, "(%.f, %.f, %.f)" % (cX,cY,theta),
                (orig.shape[1] - 200, orig.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                 0.65, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Image", orig)
        if cv2.waitKey(0) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
