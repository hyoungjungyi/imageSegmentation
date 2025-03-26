#ch9.모폴로지 - 이미지 구조와 형태 분석 방법론
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
"""
Erosion, Dilation
침식연산: 커널 옮겨가며 다 올려둘 수 있으면 1, 다 올려둘 수 없으면 0
    -> 어두운 부분 노이즈 제거
팽창연산: 커널 옮겨가며 다 올려둘 수 없으면 1 로 변경
    -> 밝은 부분 노이즈 제거
"""
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\morph_dot.png")
#커널 만들어서 침식연산
k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erosion=cv2.erode(img,k)
merged=np.hstack((img,erosion))
cv2.imwrite("./7_results/erosion.jpg",merged)

img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\morph_hole.png")
k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dst=cv2.dilate(img,k)
merged=np.hstack((img,dst))
cv2.imwrite("./7_results/dilation.jpg",merged)

"""
Opening, Closing
Opening: 침식 -> 팽창
    밝은 노이즈 제거
Closing: 팽창 -> 침식
    어두운 노이즈 제거
그래디언트: 팽창 - 침식
"""
#opening
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\morph_dot.png")
k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
opening=cv2.erode(img,k)
opening=cv2.dilate(opening,k)
merged=np.hstack((img,opening))
cv2.imwrite("./7_results/opening.jpg",merged)

#closing
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\morph_hole.png")
k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
closing=cv2.dilate(img,k)
closing=cv2.erode(closing,k)
merged=np.hstack((img,closing))
cv2.imwrite("./7_results/closing.jpg",merged)

#edge detection using gradient 
"""
dilation, erosion 구현
    패딩 크기: 커널 크기의 절반, 이미지주변에 패딩을 추가. 
    result=입력 이미지와 동일한 크기,데이터타입을 가지는 0으로 채워진 배열
    ->result의 각 픽셀을 돌면서 커널과 픽셀값을 곱한 후 최대값을 찾아 저장
"""
def dilate(image, kernel):
    h,w=image.shape[:2]
    kh,kw=kernel.shape[:2]
    pad_h,pad_w=kh//2,kw//2
    #패딩을 0으로 해야 최대값 연산에 영상 없음
    padded=cv2.copyMakeBorder(image,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=0)
    result=np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            result[i,j]=np.max(padded[i:i+kh,j:j+kw]*kernel)
            
    return result
def erode(image, kernel):
    h,w=image.shape[:2]
    kh,kw=kernel.shape[:2]
    pad_h,pad_w=kh//2,kw//2
    #패딩을 255로 해야 최솟값 연산에 영향 없음 
    padded=cv2.copyMakeBorder(image,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=255)
    result=np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            result[i,j]=np.min(padded[i:i+kh,j:j+kw]*kernel)
    return result
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\morphological.png")
k=np.ones((3,3),np.uint8)
dilated=dilate(img,k)
eroded=erode(img,k)
gradient=cv2.subtract(dilated,eroded)
merged=np.hstack((img,dilated,eroded,gradient))
cv2.imwrite("./7_results/gradient.jpg",merged)