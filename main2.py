import cv2
import numpy as np
import dlib
from imutils import face_utils, resize

# 과일 이미지 크기 조정
apple_img = cv2.imread('apple.jpeg')
apple_img = cv2.resize(apple_img, dsize=(512, 512))
# 데이터 셋을 가져오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 카메라를 열기
# cap = cv2.VideoCapture(0)
# 동영상을 열기
cap = cv2.VideoCapture('01.mp4')

while cap.isOpened:
    ret, img = cap.read()

    if not ret:
        break
    faces = detector(img)

    result = apple_img.copy()

    # 얼굴 인식
    if len(faces) > 0 :
        face = faces[0]
        x1,y1,x2,y2= face.left(), face.top(),face.right(),face.bottom()
        # 얼굴만 복사하기(crop)
        face_img = img[y1:y2, x1:x2].copy()

        shape = predictor(img,face)
        shape = face_utils.shape_to_np(shape) 

        for p in shape:
            cv2.circle(face_img, center=(p[0]-x1, p[1]-y1), radius=2, color=255,thickness=-1)

        #cv2.imshow('face',face_img)

        #if cv2.waitkey(1) == ord('q'):
        #   break

        # 눈
        # 왼눈 x축 36~39 y축 37~41
        le_x1 = shape[36,0]
        le_y1 = shape[37,1]
        le_x2 = shape[39,0]
        le_y2 = shape[41,1]
        # 마진을 주기
        # 오른눈 x축 36~39 y축 37~41
        le_margin = int((le_x2 - le_x1) * 0.18)
        re_x1 = shape[42,0]
        re_y1 = shape[43,1]
        re_x2 = shape[45,0]
        re_y2 = shape[47,1]
        # 마진을 주기
        re_margin = int((re_x2 - re_x1) * 0.18)

        # crop
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin,le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin,re_x1-re_margin:re_x2+re_margin].copy()

        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        # 합성 (poison blending)을 티안나게 한다.
        result = cv2.seamlessClone(
            left_eye_img,
            result,
            np.full(left_eye_img.shape[:2],255, left_eye_img.dtype),
            (150,200),
            # 알아서 해
            cv2.MIXED_CLONE
        )
        result = cv2.seamlessClone(
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2],255, right_eye_img.dtype),
            (300,200),
            # 알아서 해
            cv2.MIXED_CLONE
        )
        # 눈실행
        #cv2.imshow('left',left_eye_img)
        #cv2.imshow('right',right_eye_img)
        #if cv2.waitkey(1) == ord('q'):
        #    break

        # 입
        mouth_x1 = shape[48,0]
        mouth_y1 = shape[50,1]
        mouth_x2 = shape[54,0]
        mouth_y2 = shape[57,1]
        mouth_margin= int((mouth_x2-mouth_x1)*0.1)

        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin,mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2],255,mouth_img.dtype),
            (230,360),
            cv2.MIXED_CLONE
        )
#입실행
        cv2.imshow('result',result)
        if cv2.waitKey(1) == ord('q'):
            break