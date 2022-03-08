import numpy as np
import cv2


    
def lena_upside_down():

    img = cv2.imread('lena.bmp') #read image
    #cv2.imshow("My Img", img) #shoe the image and test
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    udimg = np.zeros(shape=(imgWidth,imgHeight,3))

    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            udimg[i][j]= img[imgHeight-i-1][j]

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    udimg = udimg.astype(np.uint8) #set the right data type
    # cv2.imshow("output Img", udimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("upside_down_lena.bmp", udimg) #write the output of image

def lena_right_side_left():
    img = cv2.imread('lena.bmp') #read image
    
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    rlimg = np.zeros(shape=(imgWidth,imgHeight,3))

    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            rlimg[i][j]= img[i][imgWidth-j-1]

    rlimg = rlimg.astype(np.uint8) #set the right data type
    cv2.imwrite("right_side_left_lena.bmp", rlimg) #write the output of image

def lena_daigonally_flip():
    img = cv2.imread('lena.bmp') #read image
 
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    diaimg = np.zeros(shape=(imgWidth,imgHeight,3))

    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            diaimg[i][j]= img[j][i]

    diaimg = diaimg.astype(np.uint8) #set the right data type
    cv2.imwrite("diagonally_flip_lena.bmp", diaimg) #write the output of image


def lena_rotate(angle, center=None, scale=1.0):
    img = cv2.imread('lena.bmp') #read image
    (imgHeight, imgWidth) = img.shape[:2]

    if center is None: ## set center as image center if not define
        center = (imgWidth / 2, imgHeight / 2)
    # rotate matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (imgWidth, imgHeight))
    cv2.imwrite("45_rotate_lena.bmp", rotated) #write the output of image

def lena_resize(rate):
    img = cv2.imread('lena.bmp') #read image
    (imgHeight, imgWidth) = img.shape[:2]
    dim = (int(imgWidth * rate), int(imgHeight * rate))

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("resized_lena.bmp", resized) #write the output of image

def binarized_lena():

    img = cv2.imread('lena.bmp') #read image
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    binimg = np.zeros(shape=(imgWidth,imgHeight,3))
    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            if  img[i][j].mean() >=128:
                binimg[i][j]= np.array([255,255,255])
            else:
                binimg[i][j]= np.array([0,0,0])

    binimg = binimg.astype(np.uint8) #set the right data type
    cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image

def main():
    lena_upside_down()
    lena_right_side_left()
    lena_daigonally_flip()
    lena_rotate(-45)
    lena_resize(0.5)
    binarized_lena()

if __name__ == '__main__':
    main()