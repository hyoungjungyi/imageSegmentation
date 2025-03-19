#ch5-영상 필터링
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
"""
HPF high pass filter: 고주파만 통과시킴
    에지검출에 주로 쓰임
LPF low pass filter: 저주파만 통과시킴
    고주파 노이즈 제거에 쓰임
"""
gray=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna_gray.png",cv2.IMREAD_GRAYSCALE) 
height, width= gray.shape
#이산 푸리에 변환 수행하여 복소수 형태 출력 생성, 푸리에 변환 결과 저장 -> 푸리에 변환이 뭐지
dft=cv2.dft(np.float32(gray),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift=np.fft.fftshift(dft)
row, col=int(height/2), int(width/2)
LPF=np.zeros((height,width,2),np.uint8)
LPF[row-50:row+50,col-50:col+50]=1
LPF_shift=dft_shift*LPF
LPF_ishift=np.fft.ifftshift(LPF_shift)
LPF_img=cv2.idft(LPF_ishift)
LPF_img=cv2.magnitude(LPF_img[:,:,0],LPF_img[:,:,1])
out=20*np.log(cv2.magnitude(LPF_shift[:, :,0],LPF_shift[:, :,1]))

HPF=np.ones((height,width,2),np.uint8)
HPF[row-50:row+50,col-50:col+50]=0
HPF_shift=dft_shift*HPF
HPF_ishift=np.fft.ifftshift(HPF_shift)
HPF_img=cv2.idft(HPF_ishift)
HPF_img=cv2.magnitude(HPF_img[:,:,0],HPF_img[:,:,1])
out2=20*np.log(cv2.magnitude(HPF_shift[:, :,0],HPF_shift[:, :,1]))


plt.subplot(151), plt.imshow(gray, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152), plt.imshow(LPF_img, cmap='gray')
#영상이 부드러워지고 블러링된 형태
plt.title('LPF'), plt.xticks([]), plt.yticks([])
#주파수 영역 시각화 > 중심부만 밝게(저주파 통과 확인)
plt.subplot(153), plt.imshow(out, cmap='gray')
plt.title('out1'), plt.xticks([]), plt.yticks([])
plt.subplot(154), plt.imshow(HPF_img, cmap='gray')
#중심부 영역을 차단해서 저주파 성분 제거 > 엣지 강조
plt.title('HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(155), plt.imshow(out2, cmap='gray')
#주파수 영역 시각화 > 중심부 어둡게(저주파 차단 확인)
plt.title('out2'), plt.xticks([]), plt.yticks([])
plt.savefig("./5_results/LPF_HPF.png")

"""
convolution
    -blurring filter: 미디안 블러링 / 가우시안 블러링
        가우시안 블러링: 중앙일수록 가중치가 높고 멀어질수록 가중치가 작아짐
            밀도가 동일한 노이즈/백색노이즈 제거하는데 효과적
    -sharpening filter
"""
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\lenna.png")

kernel=np.ones((5,5))/5**2
blurred=cv2.filter2D(img,-1,kernel)
cv2.imwrite("./5_results/median blur.png",blurred)
#커널 직접 생성(바깥일수록 작아지는 구조)
k1=np.array([[1,2,1],
             [2,4,2]
             [1,2,1]])*(1/16)
blur1=cv2.filter2D(img,-1,k1)
#커널을 api로 얻어서 블러링
k2=cv2.getGaussianKernel(3,0)
blur2=cv2.filter2D(img,-1,k2*k2.T)
#가우시간 블러 api로 블러링
blur3=cv2.GaussianBlur(img,(3,3),0)
imgs={'Original':img,'kernel':blur1,'kernelApi':blur2,"apiBlur":blur3}
cv2.imwrite("./5_results/kernel.png",blur1)
cv2.imwrite("./5_results/kernelApi.png",blur2)
cv2.imwrite("./5_results/blurApi.png",blur3)

#샤프닝
mask=np.array([[-1,-1,-1],
               [-1,9,-1],
               [-1,-1,-1]],dtype=np.float32)
sharpening_img=cv2.filter2D(img,-1,mask)
cv2.imwrite("./5_results/sharpening.png",sharpening_img)