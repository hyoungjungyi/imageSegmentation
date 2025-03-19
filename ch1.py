#ch1 - 영상 기본
import cv2
import sys
import numpy as np

"""
컬러&흑백 영상 입출력, 색상 채널 분리
"""
# 흑백이미지 읽기
image = cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lenna_gray.png", 0) 
image1=cv2.imwrite("./1_results/bnw_input.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 색상 이미지 읽기
color_img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lena.jpg", cv2.IMREAD_COLOR)
image2=cv2.imwrite("./1_results/color_input.jpg", color_img)


"""
색 추출과 변환 ( GRAY, HSV, RGB, CMY, HSI)
"""
image = cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lena.jpg")
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#hsv영상을 단일 채널별로 분리
h,s,v=cv2.split(hsv)
#파란색 영역만 WHITE로 이진화
h_red=cv2.inRange(h,130,170)
#파란색 영역만 이진화된 영상을 마스크로 써서 BITWISE AND연산
dst=cv2.bitwise_and(hsv,hsv,mask=h_red)
#HSV영상을 BGR영상으로 변환
dst=cv2.cvtColor(dst,cv2.COLOR_HSV2BGR)
image3=cv2.imwrite("./1_results/blue_split.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#색변환
gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image4=cv2.imwrite("./1_results/change_gray.jpg", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

hsv_img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
image14=cv2.imwrite("./1_results/change_hsv_h.jpg",h)
image15=cv2.imwrite("./1_results/change_hsv_s.jpg",s)
image16=cv2.imwrite("./1_results/change_hsv_v.jpg",v)
image5=cv2.imwrite("./1_results/change_hsv.jpg", hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image6=cv2.imwrite("./1_results/change_rgb.jpg", rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def bgr_to_cmy(image):
    # 0에서 255사이의 값인 BGR을 0-1 실수로 정규화(for계산 단순화)
    bgrdash = image.astype(float) / 255.0
    #K 블랙값계산
    K=1-np.max(bgrdash, axis=-1) 
    # cmy는 빛을 흡수하므로 1에서 rgb값을 뺌
    C = (1 - bgrdash[..., 2]-K)/(1-K)
    M = (1 - bgrdash[..., 1]-K)/(1-K)
    Y = (1 - bgrdash[..., 0]-K)/(1-K)
    # 다시 0에서 255사이의 값으로 반환
    C = (C * 255).astype(np.uint8)
    M = (M * 255).astype(np.uint8)
    Y = (Y * 255).astype(np.uint8)
    return (C, M, Y)
# Convert the image to CMY
C, M, Y = bgr_to_cmy(image)
# Save the CMY channels as separate images
image7=cv2.imwrite("./1_results/change_cmy_c.jpg", C)
image8=cv2.imwrite("./1_results/change_cmy_m.jpg", M)
image9=cv2.imwrite("./1_results/change_cmy_y.jpg", Y)
cv2.waitKey(0)
cv2.destroyAllWindows()

def bgr_to_hsi(image):
    bgrdash = image.astype(float) / 255.0
    B, G, R = bgrdash[..., 0], bgrdash[..., 1], bgrdash[..., 2]
    # Intensity(명도,밝기) 계산 - rgb의 평균
    I = (R + G + B) / 3
    # Saturation(채도) 계산: 흰색 제외한 순수 색상 비율= 1-흰색이 섞인 정도 {(3/총 밝기)*가장 작은 값}
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_val
    # Hue 계산 - 분자 분모 계산(RGB의 상대적 비율), 코사인 각도 계산
    """
    num = rgb값의 상대적 차이. r이 g,b보다 얼마나 더 많은지
    den = rgb 간 상대적 차이를 정규화
    theta = 코사인 역함수로 rgb값의 차이를 각도로 변환 => 이후 hue로 변환
    -num, den 따로 구하는 이유: rgb의 상대적 비율을 정확히 반영하기 위해
    """
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # 정규화
    H = (H * 255).astype(np.uint8)
    I = (I * 255).astype(np.uint8)
    S = (S * 255).astype(np.uint8)
    return (H, I, S)
H, I, S = bgr_to_hsi(image)
image10=cv2.imwrite("./1_results/change_hsi_h.jpg", H)
image11=cv2.imwrite("./1_results/change_hsi_s.jpg", S)
image12=cv2.imwrite("./1_results/change_hsi_i.jpg", I)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
영상 픽셀 단위 접근 및 색변환
"""
img_halfgray =cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lena.jpg")
count = 0
height, width, channel = img_halfgray.shape


print( height, width, channel)   # = print img_color.shape
# 이미지의 1/4를 gray 로 변환하는 코드
for y in range(0, int(height/2)):
    for x in range(0, int(width/2)):
        b = img_halfgray.item(y, x, 0)
        g = img_halfgray.item(y, x, 1)
        r = img_halfgray.item(y, x, 2)

        gray = (int(b) + int(g) + int(r)) / 3.0

        if gray > 255:
            gray = 255

        img_halfgray[y,x]=gray



image13=cv2.imwrite("./1_results/img_halfgray.jpg", img_halfgray)
cv2.waitKey(0)
cv2.destroyAllWindows()

