import cv2
import pafy
import numpy as np

url = "https://www.youtube.com/watch?v=6wV1VC9AadA"
videoInfo = pafy.new(url)
best = videoInfo.getbest(preftype='mp4')

videoPath = best.url

cap = cv2.VideoCapture(videoPath)

fps = cap.get(cv2.CAP_PROP_FPS)
if cap.isOpened():
    saveFilePath = './record.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    out = cv2.VideoWriter(saveFilePath, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1= cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([130,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask3 = mask1 + mask2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        red = cv2.bitwise_and(frame,frame, mask=mask3)

        good = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        nice = good+red

        #cv2.imshow('video', gray)
        #cv2.imshow('video1',red)
        cv2.imshow('video2',nice)

        if cv2.waitKey(int(1000/fps)) >= 0:
            break
    out.release()

cap.release()
cv2.destroyAllWindows()

