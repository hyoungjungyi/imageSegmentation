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
img=cv2.GaussianBlur(img,(5,5),0.3)
Gx=cv2.Sobel(np.float32(img),cv2.CV_32F,1,0,3)
Gy=cv2.Sobel(np.float32(img),cv2.CV_32F,0,1,3)
#2 소벨 마스크 sobel: 그래디언트 강도를 담고있는 2D행렬 (Gx Gy 합친것)
sobel=cv2.magnitude(Gx,Gy)
sobel=np.clip(sobel,0,255).astype(np.uint8)
print(sobel)
#3 비최대치 억제
"""
non max suppresion
에지 픽셀 중 최대값이 아닌 픽셀을 제거하여 엣지를 얇게 만듦
    dst: 결과를 저장할 배열
    현재 픽셀 주변 3x3 영역을 1차원 배열로 변환하고 하나씩 돌면서 비교
    현재 픽셀이 최대값이면 유지, 아니면 0
"""
def non_max_suppression(sobel, direct):
    rows, cols=sobel.shape[:2]
    dst=np.zeros((rows,cols),np.float32)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            #관심 영역 참조를 통해 이웃 화소 가져오기
            values=sobel[i-1:i+2,j-1:j+2].flatten()
            first=[3,0,1,2]
            id=first[direct[i,j]]
            v1,v2=values[id],values[8-id]
            dst[i,j]=sobel[i,j] if (v1<sobel[i,j]>v2) else 0
    return dst
directs=cv2.phase(Gx,Gy)/(np.pi/4)
directs=directs.astype(int)%4

#4 히스테리시스 임계값

#pos_ck: 픽셀이 이미 방문되었는지 기med록/canny: 엣지 여부를 기록. 둘 다 0/255
pos_ck = np.zeros(img.shape[:2], np.uint8)
canny = np.zeros(img.shape[:2], np.uint8)
max_sobel = non_max_suppression(sobel, directs)
max_sobel = max_sobel.astype(np.uint8)
print(f"<max_sobel>\n화소값 총합 : {cv2.sumElems(max_sobel)} \n화소 최대값 : {np.max(max_sobel)} \n화소 최소값 : {np.min(max_sobel)} \n행렬 형태 : {max_sobel.shape}")


print(sobel >= max_sobel)
checker = sobel >= max_sobel
unique, counts = np.unique(checker, return_counts=True)
checker = dict(zip(unique, counts))
print(checker)

m = 0
n = 0
print(f"sobel의 화소값 : {sobel[m, n]} \nmax_sobel의 화소값 : {max_sobel[m, n]}")

"""
trace, hysteresis thresholding
trace: 재귀적으로 이웃픽셀을 추적하여 강한 엣지와 연결된 약한 엣지를 찾음
hystheresis_th: 상위 임계값을 넘는 픽셀에서 시작하여 연결된 약한 엣지를 추적
-> 약한 엣지를 노이즈일 수 있지만 강한 엣지와 연결된 애들만 포함시켜서 정확도를 높임
-> hystheresis_th에서 강한 엣지를 발견하면 trace함수로 약한 엣지를 추적.
"""
def trace(max_sobel,i,j,low):
    h,w=max_sobel.shape
    if (0<=i<h and 0<=j<w)==False:return
    if pos_ck[i,j]>0 and max_sobel[i,j]>=low:
        pos_ck[i,j]=255
        canny[i,j]=255
        trace(max_sobel,i-1,j-1,low)
        trace(max_sobel,i,j-1,low)
        trace(max_sobel,i+1,j-1,low)
        trace(max_sobel,i-1,j,low)
        trace(max_sobel,i+1,j,low )
        trace(max_sobel,i-1,j+1,low)
        trace(max_sobel,i,j+1,low)
        trace(max_sobel,i+1,j+1,low)
def hysteresis_th(max_sobel,low,high):
    rows,cols=max_sobel.shape[:2]
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if max_sobel[i,j]>=high:
                trace(max_sobel,i,j,low)
            
nonmax = max_sobel.copy()


hysteresis_th(max_sobel, 100, 150)

print(nonmax)
print(max_sobel)
print(nonmax == max_sobel)

canny = max_sobel.copy()
canny2 = cv2.Canny(img, 100, 150)


cv2.imwrite("./6_results/sobel.jpg", sobel)
cv2.imwrite("./6_results/canny.jpg", canny)
cv2.imwrite("./6_results/OpenCV_Canny.jpg", canny2)
cv2.waitKey(0)

#sharpening:원본 영상에 엣지영상을 더해서 더 명확하게 만듦
sharpen=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png")
sharpen[canny==255]=(0,0,255)

cv2.imwrite("./6_results/edge_highlight.jpg",sharpen)

"""
Hough Transform
    lines_standard: 일반적인 직선 검출
    lines_standard 안을 돌면서 허프변환결과가 존재하면 그 직선의 파라미터를 추출
"""
hough_img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\sudoku.png")
hough_g=cv2.cvtColor(hough_img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(hough_g,100,200)
lines_standard=cv2.HoughLines(edges,1,np.pi/180,132)

if lines_standard is not None:
    for line in lines_standard:
        rho, theta=line[0]
        cos, sin = np.cos(theta), np.sin(theta)
        cx, cy = rho * cos, rho * sin
        x1, y1 = int(cx + 1000 * (-sin)), int(cy + 1000 * cos)
        x2, y2 = int(cx + 1000 * sin), int(cy + 1000 * (-cos))
        # 원본 사진에 초록색 선으로 표시
        cv2.line(hough_img, (x1, y1), (x2, y2), (0,255,0), 1)
  
cv2.imwrite("./6_results/hough_img.jpg",hough_img)