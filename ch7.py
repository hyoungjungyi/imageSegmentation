#ch7.주파수 영역과 노이즈
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
"""
-푸리에 변환 FT
    연속 시간영역 신호 -> 주파수 영역 연속함수, 주로 이론적 분석에 사용
-이산 푸리에 변환 DFT
    이산 시간 신호 -> 이산 주파수 신호
-고속 푸리에 변환 FFT
    DFT의 시간 복잡도 개선 버젼
-이산 코사인 변환 DCT
    이산신호 -> 코사인 함수의 합, 주로 이미지.오디오 압축에 사용
    
magnitude: 푸리에 변환 결과에서 복소수 형태의 주파수 성분 크기. 이미지 주파수 성분 강도라는 뜻
210->30: 주파수를 약화시켜서 이미지를 부드럽게만듦
반경 45 픽셀영역 기준의 이유: 푸리에 변환해서 shift하면 고주파가 가장자리, 저주파가 중심에 위치하므로
    반경 45 픽셀을 기준으로 하면 저주파 영역은 중앙에, 고주파 영역은 가장자리에 위치하게 됨
    따라서 고주파 영역을 돌면서 너무 높으면 낮추는거임
"""




img = cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\Original_Image.png", cv2.IMREAD_GRAYSCALE)

# dft 변환하고 shift 해서 저주파를 중앙으로 이동
# magnitude 계산하고 로그 변환
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# 주어진 점이 중심으로부터 특정 반경 내에 있는지 확인하는 함수
def dis(a, b, radius):
    tmp = (magnitude_spectrum.shape[0] // 2 - a) ** 2 + (magnitude_spectrum.shape[1] // 2 - b) ** 2
    tmp = round(math.sqrt(tmp))
    if tmp <= radius:
        return True
    else:
        return False



# 마스크 생성
rows, cols = img.shape
mask = np.zeros((rows,cols,1),np.uint8)

# 이중 for문 돌면서 중심에서 반경 45 픽셀 밖의 영역에서 magnitude 가 210 이상이면 30으로 변경 -> 마스크에 해당 값 저장
for i in range(len(magnitude_spectrum)):
    for j in range(len(magnitude_spectrum[0])):
        if not dis(i, j, 45):
            if magnitude_spectrum[i][j] >= 210:
                magnitude_spectrum[i][j] = 30
        mask[i][j] = min(magnitude_spectrum[i][j], 255)


# 원본 dft결과에 마스크 적용 -> 노이즈 제거
fshift = (dft_shift * mask)

# shift된 dft결과를 원위치로 되돌림 / 역dft 적용하여 이미지 도메인으로 변환 / 복소수 결과 계산하여 최종 이미지 얻
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


def show_results(image, spectrum, mask, result):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(spectrum, cmap='gray')
    axs[1].set_title("Spectrum Image")
    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title("Mask")
    axs[3].imshow(result, cmap='gray')
    axs[3].set_title("Result Image")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("./7_results/dft_result.png")
    plt.show()

show_results(img,magnitude_spectrum, mask, img_back)

