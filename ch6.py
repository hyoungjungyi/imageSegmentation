#ch6-엣지 검출
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

"""
sobel mask
    sobel_x: 수평 방향 밝기 변화
    sobel_y: 수직 방향 밝기 변화
    sobel_combined: 두 개를 합쳐서 모든 방향의 엣지를 검출
"""
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def apply_sobel(image):
    #cv2.Sobel(이미지, 출력 데이터타입, x축 방향 미분차수, y축 방향 미분차수, 커널의 크기)
    sobel_x=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobel_y=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    sobel_combined=cv2.magnitude(sobel_x,sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sobel_x, sobel_y, sobel_combined
def show_results(image, sobel_x, sobel_y,sobel_combined):
    plt.figure(figsize=(10,5))
    plt.subplot(1,4,1)
    plt.imshow(image,cmap='gray')
    plt.title("Original Image")
    plt.subplot(1,4,2)
    plt.imshow(sobel_x,cmap='gray')
    plt.title("Sobel X")
    plt.subplot(1,4,3)
    plt.imshow(sobel_y,cmap='gray')
    plt.title("Sobel Y")
    plt.subplot(1,4,4)
    plt.imshow(sobel_combined,cmap='gray')
    plt.title("Sobel Combined")

sobel_x, sobel_y, sobel_combined=apply_sobel(img)
show_results(img, sobel_x,sobel_y,sobel_combined)
plt.savefig("./6_results/sobel_mask.jpg")

"""
라플라시안 검출
    라플라시안: 가우시안으로 스무딩>라플라시안으로 이차미분>0교차로 엣지검출
    캐니: 가우시안 스무딩>A마스크 사용해서 그라디언트 영상 계산>그라디언트 영상에서 에지의 방향,강도 계산>비최대 억제로 최대한 가는 엣지를 검출
    그라디언트 방향: 어두운 곳 -> 밝은 곳
    에지 화소로 살아남은 화소들: 그라디언트 방향으로 극대값을 갖는 애들
    Double thresholding으로 엣지 구분: non edge / weak edge / strong edge
"""
#그냥 라플라시안 검출
def apply_laplacian(image):
    laplacian=cv2.Laplacian(image,cv2.CV_64F,ksize=1)
    return laplacian
laplacian=apply_laplacian(img)
laplacian = cv2.normalize(laplacian, None, alpha=-128, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("./6_results/laplacian.jpg",laplacian)

#가우시안 스무딩 후 라플라시안 검출
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(img,(5,5),0)
laplacian=apply_laplacian(blur)
laplacian = cv2.normalize(laplacian, None, alpha=-128, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("./6_results/GNlaplacian.jpg",laplacian)

sharp_img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\5_results\sharpening.png")
sharp_img=cv2.cvtColor(sharp_img,cv2.COLOR_BGR2GRAY)
sharp_img=cv2.GaussianBlur(sharp_img,(5,5),0)
sharp_laplacian=apply_laplacian(sharp_img)
sharp_laplacian = cv2.normalize(sharp_laplacian, None, alpha=-128, beta=128, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite("./6_results/sharpened_laplacian.jpg",sharp_laplacian)

"""
canny edge 검출 단계
    1. 가우시안 필터링
    2. sobel mask 사용해서 x,y방향의 gradient 계산
    3. non maximum suppresion() > 가장 가는 엣지 검출
    4. hysteresis thresholding > 엣지 결정
"""
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#1 가우시안 필터링
img=cv2.GaussianBlur(img,(5,5),0)
Gx=cv2.Sobel(np.float32(img),cv2.CV_32F,1,0,3)
Gy=cv2.Sobel(np.float32(img),cv2.CV_32F,0,1,3)
#2 소벨 마스크
sobel=cv2.magnitude(Gx,Gy)
sobel=np.clip(sobel,0,255).astype(np.uint8)
cv2.imshow("sobel",sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
#3 비최대치 억제
def non_max_suppression(sobel, direct):
    rows, cols=sobel.shape[:2]
    dst=np.zeros((rows,cols),np.float32)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            #관심 영역 참조를 통해 이웃 화소 가져오기
            values=sobel[i-1:i+2,j-1:j+2].flatten()
            first=[3,0,1,2]
            