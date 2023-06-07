# 요구사항
#유튜브에서 특정 영상을 다운받아 기록하는 것을 기반
#1.유튜브영상을 스케치로 변경하여 저장
#2.유튜브영상내에 특정컬러 추적하여 필터 후 저장
#   2-1. 특정컬러의 영역에 사각테두리를 둘러서 표시
#   2-2. 특정컬러의 영역만 마스킹하여 해당 컬러의 이미지만 색상이 있도록 (배경은 흑백) 

#사용기술
# pafy or cap_from_youtube
#opencv
#hsv 컬러맵에 대한 이해
# 스케치 함수 사용에 대한 이해 (이전 코드를 참고)

import cv2
import pafy

url = "https://www.youtube.com/watch?v=H5kokc0ULuo"
videoInfo = pafy.new(url)
best = videoInfo.getbest(preftype='mp4')

videoPath = best.url
cap = cv2.VideoCapture(videoPath)

outputFilePath = "jump.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(outputFilePath, fourcc, fps, (width, height), isColor=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    sketch_gray, sketch_color = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    sketch = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    
    out.write(sketch)
    
    cv2.imshow('Sketch Video', sketch)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
