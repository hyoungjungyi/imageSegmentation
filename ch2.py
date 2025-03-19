#ch2 - 영상 변환
import cv2
import numpy as np

"""
업샘플링 다운샘플링 (해상도 변경)
"""
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\lenna\lena.jpg") 
height, width, channel= img.shape
dst=cv2.pyrUp(img, dstsize=(width*2,height*2), borderType=cv2.BORDER_DEFAULT)
image1=cv2.imwrite("./2_results/upsampling.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst=cv2.pyrDown(img, dstsize=(int(width/2),int(height/2)), borderType=cv2.BORDER_DEFAULT)
image2=cv2.imwrite("./2_results/downsampling.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
이미지 크기 확대축소/대칭회전
"""
#확대축소
dst=cv2.resize(img, dsize=(int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA)
image3=cv2.imwrite("./2_results/resizing_down.jpg", dst)
dst=cv2.resize(img, dsize=(int(width*2.5), int(height*2.5)), interpolation=cv2.INTER_AREA)
image4=cv2.imwrite("./2_results/resizing_up.jpg", dst)
#대칭
dst=cv2.flip(img, flipCode=1)
image5=cv2.imwrite("./2_results/flip_y.jpg", dst)
dst=cv2.flip(img, flipCode=0)
image6=cv2.imwrite("./2_results/flip_x.jpg", dst)
dst=cv2.flip(img, flipCode=-1)
image7=cv2.imwrite("./2_results/flip_xy.jpg", dst)
#회전: 회전 행렬 dst 구하고 > 그 행렬을 이미지에 적용
dst=cv2.getRotationMatrix2D(center=(width/2, height/2), angle=90, scale=1.0)
rotated_image = cv2.warpAffine(img, dst, (width, height))
cv2.imwrite("./2_results/rotate_90.jpg", rotated_image)

"""
이미지 패딩-정사각형으로 만들기
"""
#가로세로 길이 차를 dif로 구한뒤에 top.bottom /  left.right에 픽셀 몇개씩 채워야하는지 확인하기
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\ct_data\0.png")
background_color=(255,0,0)
def expand2square(image, background_color):
    height,width,channel=image.shape
    if width==height:
        return image
    elif width>height:
        dif=width-height
        top=dif//2
        bottom=dif-top
        image=cv2.copyMakeBorder(image,top,bottom,0,0,cv2.BORDER_CONSTANT,value=background_color)
        return image
    else:
        dif=height-width
        left=dif//2
        right=dif-left
        image=cv2.copyMakeBorder(image,0,0,left,right,cv2.BORDER_CONSTANT,value=background_color)
        return image
image8=expand2square(img, background_color)
cv2.imwrite("./2_results/padding.jpg", image8)

"""
affine변환, 투시변환
"""
#어파인 변환
img=cv2.imread(r"C:\Users\DoyoungJang\Desktop\hjwork\videocomparison\2_results\resizing_down.jpg")
pts1=np.float32([[0,0],[0,height],[width,0]])
pts2=np.float32([[100,50],[10,height-50],[width-100,0]])
M=cv2.getAffineTransform(pts1, pts2)
affine_transformed=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
image9=cv2.imwrite("./2_results/affine.jpg", affine_transformed)
#투시 변환
srcQuad=np.float32([[0,0],[width,0],[0,height],[width,height]])
dstQuad = np.float32([[100, 50], [width - 100, 50], [10, height - 50], [width - 10, height - 50]])
pers=cv2.getPerspectiveTransform(srcQuad,dstQuad)
dst=cv2.warpPerspective(img, pers, (width, height))
image10=cv2.imwrite("./2_results/perspective.jpg", dst)