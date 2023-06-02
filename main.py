import cv2, dlib, os
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils



detector =dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')


img_path = 'img/5.jpg'
filename, ext = os.path.splitext(os.path.basename(img_path))
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dets = detector(img, upsample_num_times=1)
print(dets)
img_result = img.copy()

for i,d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}"
        .format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
    
    x1, y1 = d.rect.left(),d.rect.top()
    x2, y2 = d.rect.right(),d.rect.bottom()

    cv2.rectangle(img_result, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
shapes = []

for i,d in enumerate(dets):
    shape = predictor(img, d.rect)
    shape = face_utils.shape_to_np(shape)

    for i, p in enumerate(shape):
        shapes.append(shape)
        cv2.circle(img_result,center=tuple(p), radius=3, color=(0,0,255), thickness=-1,lineType=cv2.LINE_AA)
        cv2.putText(img_result,str(i),tuple(p), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA) 
img_out = cv2.cvtColor(img_result,cv2.COLOR_RGB2BGR)

from math import atan2, degrees

def overlay_transparent(background_img, img_to_overlay_t,x,y,overlay_size=None):
    img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_BGR2RGBA)
    bg_img = background_img.copy()
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img,cv2.COLOR_RGB2RGBA)
    
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a,5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2): int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_bg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t,mask=mask)

    bg_img[int(y-h/2): int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg,img2_bg)
    bg_img =cv2.cvtColor(bg_img,cv2.COLOR_RGBA2RGB)

    return bg_img






cv2.imwrite('img/%s_out%s' % (filename,ext), img_out)
plt.figure(figsize=(16,16))
plt.imshow(img_result)
plt.show()
