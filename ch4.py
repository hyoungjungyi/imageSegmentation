#ch4 - 이진화
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
"""
thresholding: 임계값을 넘지 못하는 픽셀/넘는 픽셀로 이진화
hysteresis thresholding은 엣지 검출에도 사용(픽셀 자신만 보는게 아니라 주변의 분류결과까지 반영)
    thresh binary: 0/1
    thresh binary inv: thresh binary의 반대
    thresh trunc: 임계값 넘으면 value주고 안 넘으면 유지
    thresh tozero: 임계값 넘으면 0, 안 넘으면 유지
    thresh tozero inv: thresh tozero의 반대

"""
#thresholding
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna_gray.png",cv2.IMREAD_GRAYSCALE)
thresh_np=np.zeros_like(img)
thresh_np[img>127]=255
ret, thresh_cv=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(ret)
imgs={"Original":img, 'Numpy API': thresh_np,'cv2.threshold': thresh_cv}
for i,(key,value) in enumerate (imgs.items()):
    plt.subplot(1,3,i+1)
    plt.title(key)
    plt.imshow(value,cmap='gray')
    plt.xticks([]),plt.yticks([])
plt.savefig("./3,4_results/thresholding.jpg")  
#여러 thresholding flag비교
_,t_bin =cv2.threshold(img,127,255,cv2.THRESH_BINARY)
_,tbinv=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
_,t_truc=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
_,t_2zr=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
_,t_2zrinv=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
imgs={'Original':img,'Binary':t_bin,'Binary Inverse':tbinv,'Truncate':t_truc,'To Zero':t_2zr,'To Zero Inverse':t_2zrinv}
plt.figure(figsize=(12,8))
for i,(key,value) in enumerate(imgs.items()):
    plt.subplot(2,3,i+1)
    plt.title(key)
    plt.imshow(value,cmap='gray')
    plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.savefig("./3,4_results/thresholding_flag.jpg")

"""
오츠의 이진화 알고리즘: 임계값을 한번에 찾음
임계값을 임의로 정해 픽셀을 두 부류로 나누고 두 부류의 명암 분포를 구하는 작업을 반복 
-> 두 부류의 명암 분포가 가장 균일할 때의 임계값을 선택
"""
#otsu
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\ct_data\4.png",cv2.IMREAD_GRAYSCALE)
#threshold를 130으로 지정
_,t_130=cv2.threshold(img,130,255,cv2.THRESH_BINARY)
#otsu
t, t_otsu=cv2.threshold(img,130,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('otsu threshold',t)
imgs={'Original':img,'Threshold 130':t_130,'Otsu':t_otsu}
plt.figure(figsize=(12,8))
for i,(key,value) in enumerate(imgs.items()):
    plt.subplot(1,3,i+1)
    plt.title(key)
    plt.imshow(value,cmap='gray')
    plt.xticks([]),plt.yticks([])
plt.savefig("./3,4_results/otsu.jpg")

#adaptive thresholding: 픽셀 주변의 영역에 대해 임계값을 계산
blk_size=9
C=5
#오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret,th1=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#어댑티드 쓰레시홀드를 평균과 가우시간 분포로 적용
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blk_size,C)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blk_size,C)
plt.figure(figsize=(12,8))
imgs={'Original':img, 'Global Otsu':th1,'Adaptive Mean':th2,'Adaptive Gaussian':th3}
for i,(key,value) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(key)
    plt.imshow(value,'gray')
    plt.xticks([]),plt.yticks([])
plt.savefig("./3,4_results/adaptive_thresholding.jpg")
#조명 그림자차이 때문에 보통 적응형을 많이 씀