import dlib
import cv2
import os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')
img_path = 'img/18.jpg'
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
dets = detector(img, upsample_num_times=1)
img_result = img.copy()

for i, d in enumerate(dets):
    rect = d.rect
    cv2.rectangle(img_result, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)
    shape = predictor(img, rect)
    shape = face_utils.shape_to_np(shape)
    for i, p in enumerate(shape):
        cv2.circle(img_result, tuple(p), 3, (0, 0, 255), -1)
        cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

horns = cv2.imread('img/horns2.png', cv2.IMREAD_UNCHANGED)
nose = cv2.imread('img/nose.png', cv2.IMREAD_UNCHANGED)
img_result2 = img.copy()

for shape in shape:
    if len(shape) >= 5:
        horns_center = np.mean([shape[4], shape[1]], axis=0) // [1, 1.3]
        horns_size = np.linalg.norm(shape[4] - shape[1]) * 3
        nose_center = shape[3]
        nose_size = horns_size // 4

        angle = -np.degrees(np.arctan2(shape[1][1] - shape[4][1], shape[1][0] - shape[4][0]))
        rotated_horns = cv2.warpAffine(horns, cv2.getRotationMatrix2D((horns.shape[1] / 2, horns.shape[0] / 2), angle, 1),
                                       (horns.shape[1], horns.shape[0]))

        img_result2[nose_center[1] - int(nose_size / 2):nose_center[1] + int(nose_size / 2),
        nose_center[0] - int(nose_size / 2):nose_center[0] + int(nose_size / 2)] = cv2.resize(nose, (nose_size, nose_size))
        img_result2[horns_center[1] - int(rotated_horns.shape[0] / 2):horns_center[1] + int(rotated_horns.shape[0] / 2),
        horns_center[0] - int(horns_size / 2):horns_center[0] + int(horns_size / 2)] = cv2.resize(rotated_horns,
                                                                                                   (int(horns_size),
                                                                                                    int(
                                                                                                        horns_size * horns.shape[
                                                                                                            0] / horns.shape[
                                                                                                            1])))

plt.figure(figsize=(16, 16))
plt.imshow(img_result2)
plt.show()
