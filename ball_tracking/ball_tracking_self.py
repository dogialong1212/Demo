# USAGE
# python ball_tracking_self.py --video ball_tracking_example.mp4
# python ball_tracking_self.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

#Xây dựng cấu trúc các đối số và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to video file")
ap.add_argument("-b", "--buffer", type=int, default=40,
                help="max buffer size for centre of object")
args = vars(ap.parse_args())

#Xác định ngưỡng màu của vật thể
#Ngưỡng trên ngưỡng dưới màu trong hệ HSV
colorLower = (29, 86, 6)
colorUpper = (64, 255, 255)

#Lưu max buffer
pts = deque(maxlen = args["buffer"])

#Chuyển hướng sang webcam nếu không có đường dẫn video được thêm vào
if not args.get("video", False):
        cap = VideoStream(src=0).start()

#Nếu có thi gắn video vào
else:
        cap = cv2.VideoCapture(args["video"])

#Cho thời gian để videp hoặc camera khởi động
time.sleep(1.0)

#Vong lặp
while True:
        #Lấy frame
        frame = cap.read()

        #Xử lí frame từ VideoCapture hoặc VideoStream
        frame = frame[1] if args.get("video", False) else frame

        #Nếu kết thúc video hoặc không nhận được h́nh từ frame th́ kết thúc
        if frame is None:
                break
    
        #Resize the frame, lọc ảnh, và chuyển sang hệ màu HSV
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11,11),0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #Xây dựng mask cho màu của thể
        #Sử dụng dilate và erode để loại bỏ các chấm điểm nhỏ ra khỏi mặt nạ
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        #Lấy viền mặt nạ và khởi tạo tâm
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        #Chỉ tiến hành khi có ít nhất 1 vùng bao được tim thấy
        if len(cnts) > 0:
                #tim vong bao lớn nhât trong mặt nạ
                #Computing the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x,y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                #Chỉ thực hiện chương trinh khi bán kính đủ lớn
                if radius > 10:
                          #vẽ vong bao và tâm
                          cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                          cv2.circle(frame, center, 3, (0, 0, 255), -1)

        #Cập nhật điểm (tracked points)
        pts.appendleft(center)

        #Ṿong lặp vẽ các tracked points
        for i in range(1, len(pts)):
                #Nếu điểm bị trống thi bỏ qua
                if pts[i - 1] is None or pts[i] is None:
                          continue
                #Nếu không thi tinh độ day của đường thẳng và vẽ đường nối
                thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 1.8)
                cv2.line(frame, pts[i -1], pts[i], (0, 0, 255), thickness)

        #show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #Nhấn 'q' để thoát
        if key == ord("q"):
                break

#Nếu không dùng video thi dừng camera
if not args.get("video", False):
    cap.stop()
    

#Nếu dùng video thi giải phóng
else:
        cap.release()

#Đóng tất cả các cửa sổ
cv2.destroyAllWindows()
        
