# 열화상 카메라로 찍은 이미지를 구역별로 나눠 각 구역의 평균 온도를 파악!
# 요구사항 
#1.가로를 10칸으로 나누고 하나의 칸은 정사각형 
#2. 격자로 선을 그어서 공간을 나누기
#3. 셀을 관리할 수 있는 맵핑 (row.col) 좌표로 접근해서 특정 셀에 접근
#4. 각 셀의 화소들 색상을 종합해서 평균값을 구함. 해당 평균값은 특정 기준으로 온도의 레벨(0~9)
#5. 255 - > lv 10 =>255/10
#6. 온도레벨을 해당 셀에 문자로 표시 

import cv2
import numpy as np
# 경로

img = cv2.imread('../res/fire.jpg')
row, col  = img.shape[:2]

big = np.float32([[2,0,0],
                  [0,2,0]])
big_size =(int(col*2), int(row*2))
dst1 = cv2.warpAffine(img,big,big_size)
for j in range(8):
    cv2.line(dst1, [141*j,0],[141*j,578],[255,255,255])
    for i in range(6):
        cv2.line(dst1, [0,115*i],[990,115*i],[255,255,255])
for a in range(5):
    cv2.putText(dst1,(f'{a},0'), [10,30+a*115 ], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    for b in range(7):
        cv2.putText(dst1,(f'{a},{b}'), [10+b*141,30+a*115 ], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    #for c in range(6):
       # cv2.putText(dst1,(f'2,{c}'), [10+b*141,60 ], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    #for d in range(6):
      #  cv2.putText(dst1,(f'3,{d}'), [10+b*141,90 ], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    #for f in range(6):
     #   cv2.putText(dst1,(f'4,{f}'), [10+b*141,120], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
   
 
   



#세로선
#cv2.line(dst1, (141,0 ), (141, 578), (255,255,255)) 
#cv2.line(dst1, (282,0 ), (282, 578), (255,255,255))
#cv2.line(dst1, (423,0 ), (423, 578), (255,255,255))
#cv2.line(dst1, (564,0 ), (564, 578), (255,255,255))
#cv2.line(dst1, (705,0 ), (705, 578), (255,255,255))
#cv2.line(dst1, (846,0 ), (846, 578), (255,255,255))
#cv2.line(dst1, (990,0 ), (990, 578), (255,255,255))

#가로선
#cv2.line(dst1, (0,115 ), (990, 115), (255,255,255)) 
#cv2.line(dst1, (0,231 ), (990, 231), (255,255,255))
#cv2.line(dst1, (0,346 ), (990, 346), (255,255,255))
#cv2.line(dst1, (0,462 ), (990, 462), (255,255,255))
#cv2.line(dst1, (0,578 ), (990, 578), (255,255,255))

#              (x좌표,y좌표) ( x좌표  ,y좌표), (색)


#cv2.imshow('fire',img)
cv2.imshow('big',dst1)

cv2.waitKey()
cv2.destroyAllWindows()