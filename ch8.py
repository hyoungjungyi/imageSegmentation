#ch8 노이즈
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
"""
salt and pepper
    소금후추 뿌린것같은 노이즈를 이미지에 추가함
    for data augmentation
    
다양한 노이즈 종류
- 가우시안
    정규분포를 따르는 노이즈
- 임펄스
    이미지의 픽셀이 무작위로 다른 값으로 대체됨(소금-후추)
- 라플라시안
    2차 미분 연산자로 이미지으 급격한 변화를 감지>픽셀값 조정
- uniform
    균일분포 노이즈. 동일한 확률로 노이즈가 발생
"""
def add_noise(img):
    row,col=img.shape[:2]
    number_of_pixels=random.randint(300,10000)
    for i in range(number_of_pixels):
        #랜덤 픽셀 골라서 흰색으로 칠하기
        y_coord=random.randint(0,row-1)
        x_coord=random.randint(0,col-1)
        img[y_coord][x_coord]=255
    
    number_of_pixels=random.randint(300,10000)
    for i in range(number_of_pixels):
        #랜덤 픽셀 골라서 흰색으로 칠하기
        y_coord=random.randint(0,row-1)
        x_coord=random.randint(0,col-1)
        img[y_coord][x_coord]=0
    return img

img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png",cv2.IMREAD_GRAYSCALE)
cv2.imwrite("./7_results/saltNpepper.jpg",add_noise(img))



"""
다양한 필터로 노이즈 제거
-미디언 필터
    픽셀의 중앙값(median)을 계산하여 노이즈를 제거
-하이브리드 미디언 필터
    좌상 우하 대각선의 중앙값
    우상 좌하 대각선의 중앙값
    5X5영역 전체의 중앙값
    이 세가지 의 중앙값을 최종 결과로 선택
"""

def apply_filters(img_path):
    img = cv2.imread(img_path)
    height, width, channel = img.shape

    # 일반 미디언 필터 (3x3)
    out1 = np.zeros((height + 2, width + 2, channel), dtype=np.float64)
    out1[1: 1 + height, 1: 1 + width] = img.copy().astype(np.float64)
    temp1 = out1.copy()

    # 하이브리드 미디언 필터 (5x5)
    out2 = np.zeros((height + 4, width + 4, channel), dtype=np.float64)
    out2[2: 2 + height, 2: 2 + width] = img.copy().astype(np.float64)
    temp2 = out2.copy()

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                # 일반 미디언 필터
                out1[1 + i, 1 + j, k] = np.median(temp1[i:i + 3, j:j + 3, k])

                # 하이브리드 미디언 필터
                hybrid_temp1 = np.median((temp2[i, j, k], temp2[i + 1, j + 1, k], temp2[i + 2, j + 2, k],
                                          temp2[i + 3, j + 3, k], temp2[i + 4, j + 4, k]))
                hybrid_temp2 = np.median((temp2[i + 4, j, k], temp2[i + 3, j + 1, k], temp2[i + 2, j + 2, k],
                                          temp2[i + 1, j + 3, k], temp2[i, j + 4, k]))
                hybrid_temp3 = np.median((temp2[i: i + 5, j:j + 5, k]))
                out2[2 + i, 2 + j, k] = np.median((hybrid_temp1, hybrid_temp2, hybrid_temp3))

    out1 = out1[1:1 + height, 1:1 + width].astype(np.uint8)
    out2 = out2[2:2 + height, 2:2 + width].astype(np.uint8)

    return out1, out2

    
img_path = r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\7_results\saltNpepper.jpg"
median_filtered, hybrid_median_filtered = apply_filters(img_path)


cv2.imwrite('./7_results/median_filtered.jpg', median_filtered)
cv2.imwrite('./7_results/hybrid_median_filtered.jpg', hybrid_median_filtered)

