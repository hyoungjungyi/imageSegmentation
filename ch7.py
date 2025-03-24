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
"""



# Read the image
img = cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\imageComparison\lenna\Original_Image.png", cv2.IMREAD_GRAYSCALE)

# getting dft of original image
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# fun to tell whether a point is availaible inside circle of radius 'radius'
def dis(a, b, radius):
    tmp = (magnitude_spectrum.shape[0] // 2 - a) ** 2 + (magnitude_spectrum.shape[1] // 2 - b) ** 2
    tmp = round(math.sqrt(tmp))
    if tmp <= radius:
        return True
    else:
        return False



# creating mask
rows, cols = img.shape
mask = np.zeros((rows,cols,1),np.uint8)

# getting values pixel by pixel 
# if pixel value > 210 change it to 30
for i in range(len(magnitude_spectrum)):
    for j in range(len(magnitude_spectrum[0])):
        if not dis(i, j, 45):
            if magnitude_spectrum[i][j] >= 210:
                magnitude_spectrum[i][j] = 30
        mask[i][j] = min(magnitude_spectrum[i][j], 255)


# multiplying mask and dft of original image
fshift = (dft_shift * mask)

# applying inverse dft to get modified image
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

