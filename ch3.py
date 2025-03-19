#ch3 - 히스토그램 기법
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

"""
히스토그램 분석
"""
#흑백이미지의 히스토그램
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna_gray.png") 
height, width, channel= img.shape
hist=cv2.calcHist([img],[0],None,[256],[0,256]) #0은 채널, None은 마스크, [256]은 빈도수, [0,256]은 범위
plt.figure()
plt.title("Gray Histogram")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.savefig("./3,4_results/gray_hist.jpg")
#컬러이미지의 히스토그램
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna.png") 
colors = ['b', 'g', 'r']  
bgr_planes = cv2.split(img)
plt.figure() 
plt.title("RGB Color Histogram")  
plt.xlabel("Bins")  
plt.ylabel("Pixels") 
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])  
    plt.plot(hist, color=c) 
plt.xlim([0, 256]) 
plt.legend(colors)  
plt.savefig("./3,4_results/color_hist_rgb.jpg")  
plt.close() 
"""
low saturation: 분포 영역이 한 곳에 모임
high saturation: 분포 영역이 넓게 퍼짐
equalization: 히스토그램 모양 자체가 균일하게 변함
"""
#stretching
def saturate_contrast(p,num):
    pic=p.copy()
    pic=pic.astype('int32')
    pic=np.clip(pic+(pic-128)*num,0,255).astype('uint8')
    return pic

img=saturate_contrast(img,-0.8)
colors = ['b', 'g', 'r']  
bgr_planes = cv2.split(img)
plt.figure() 
plt.title("RGB Color Histogram")  
plt.xlabel("Bins")  
plt.ylabel("Pixels") 
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])  
    plt.plot(hist, color=c) 
plt.xlim([0, 256]) 
plt.legend(colors)  
plt.savefig("./3,4_results/low_saturation.jpg")
plt.close() 

img=saturate_contrast(img,3)
colors = ['b', 'g', 'r']  
bgr_planes = cv2.split(img)
plt.figure() 
plt.title("RGB Color Histogram")  
plt.xlabel("Bins")  
plt.ylabel("Pixels") 
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])  
    plt.plot(hist, color=c) 
plt.xlim([0, 256]) 
plt.legend(colors)  
plt.savefig("./3,4_results/high_saturation.jpg")
plt.close() 
#equalization
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna_gray.png",cv2.IMREAD_GRAYSCALE)
img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dst = cv2.equalizeHist(img)
hist = cv2.calcHist([dst], [0], None, [256], [0,256])
plt.figure(figsize=(8, 10))
plt.subplot(3, 1, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
plt.subplot(3, 1, 2), plt.imshow(dst, cmap='gray'), plt.title("Equalized Image")
plt.subplot(3, 1, 3), plt.plot(hist, color='r'), plt.title("Histogram")
plt.xlim([0, 256])
plt.savefig("./3,4_results/equalization.jpg")
plt.close()
"""
gamma correction: 빛의 강도를 비선형적으로 변형함 (인간이 그렇게 인지하니까)
감마 보정이 없으면 더 어둡고 물체를 식별하기 어려워짐 
있어야 더 자연스러운 밝기 조절
gamma correction 결과
R>1: 어두워짐
R=1: 원본
R<1: 밝아짐
"""
#gamma correction
def gamma_correction(img,gamma=1.0):
    img_normalized=img/255.0
    corrected_img=np.power(img_normalized,gamma)
    corrected_img=np.uint8(np.clip(corrected_img*255,0,255))
    return corrected_img
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna.png")
gamma=2.2
corrected_img=gamma_correction(img,gamma)
cv2.imwrite("./3,4_results/gamma_corrected.jpg",corrected_img)
hist = cv2.calcHist([corrected_img], [0], None, [256], [0,256])
colors = ['b', 'g', 'r']  
bgr_planes = cv2.split(corrected_img)
plt.figure() 
plt.title("RGB Color Histogram")  
plt.xlabel("Bins")  
plt.ylabel("Pixels") 
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])  
    plt.plot(hist, color=c) 
plt.xlim([0, 256]) 
plt.legend(colors)  
plt.savefig("./3,4_results/gamma_hist.jpg")
plt.close() 