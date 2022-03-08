import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def load_lena():
    img = cv2.imread('lena.bmp') #read image
    
    # init a white canvas
    img = img.astype(np.uint8) #set the right data type
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return img


def dilation(img, kernel, k_h, k_w, centerX, centerY):      # align center to the pixel, add together and pick the max value
    # print("do dilation")
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight,3))
    for i in range(imgHeight):
        for j in range(imgWidth):
            max = np.array([0,0,0]) # temp value for every pixel in order to find max value under kernel
            for a in range(k_h):
                for b in range(k_w):
                    if kernel[a][b] ==1 and i+a-centerY < imgHeight and j+b-centerX < imgWidth and i+a-centerY >=0 and j+b-centerX >=0 :
                        if img[i+a-centerY][j+b-centerX][0] > max[0]:
                            max = img[i+a-centerY][j+b-centerX]
            nimg[i][j] =  max

    nimg = nimg.astype(np.uint8) #set the right data type
    cv2.imwrite("dilation_lena.bmp", nimg) #write the output of image

    
    # cv2.imshow("output Img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return nimg

def erosion(img,kernel, k_h,k_w, centerX,centerY):       # align center to the pixel, substract by kernel and pick minimum
    # print("do erosion")
    (imgHeight, imgWidth) = img.shape[:2]
    # represent kernel information center point and half width and height

    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight,3))
    for i in range(imgHeight):
        for j in range(imgWidth):
            min = np.array([255,255,255]) 
            for a in range(k_h):  # kernel operation
                for b in range(k_w):
                    if kernel[a][b] ==1 and i+a-centerY < imgHeight and j+b-centerX < imgWidth and i+a-centerY >=0 and j+b-centerX >=0 :
                        if img[i+a-centerY][j+b-centerX][0] < min[0]:
                            min = img[i+a-centerY][j+b-centerX]
            nimg[i][j] =  min 

    
    nimg = nimg.astype(np.uint8) #set the right data type
    cv2.imwrite("erosion_lena.bmp", nimg) #write the output of image

    return nimg
    # cv2.imshow("output Img", nimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def closeOp(img):       # dilation then erosion
    # print("do close")
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]
    d_img = dilation(img,kernel,5,5,2,2)
    c_img = erosion(d_img,kernel,5,5,2,2)
    cv2.imwrite("closing_lena.bmp", c_img) #write the output of image
      
    # cv2.imshow("output Img c", c_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def openOp(img):        # erosion then dilation
    # print("do open")
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]
    e_img = erosion(img,kernel,5,5,2,2)
    o_img = dilation(e_img,kernel,5,5,2,2,)
    cv2.imwrite("opening_lena.bmp", o_img) #write the output of image

    # cv2.imshow("output Img o", o_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def hit_and_miss(img, j_kernel, k_kernel, j_x,j_y,k_x,k_y,k_h,k_w):
    (imgHeight, imgWidth) = img.shape[:2]

    rimg = np.zeros(shape=(imgWidth,imgHeight,3))
    for i in range(imgHeight):
        for j in range(imgWidth):
            if img[i][j].mean() ==255:
                rimg[i][j] = np.array([0,0,0])
            if img[i][j].mean() ==0:
                rimg[i][j] = np.array([255,255,255])      
    rimg = rimg.astype(np.uint8) #set the right data type

    # get two results of erosion image with different kernel 
    je_img = erosion(img,j_kernel,k_h,k_w,j_x,j_y)
    ke_img = erosion(rimg,k_kernel,k_h,k_w,k_x,k_y)

    fimg = np.zeros(shape=(imgWidth,imgHeight,3))
    for i in range(imgHeight):
        for j in range(imgWidth):
            if je_img[i][j].mean() == 255 and ke_img[i][j].mean() ==255:
                fimg[i][j] = np.array([255,255,255]) 
                   
    fimg = fimg.astype(np.uint8) #set the right data type
    cv2.imwrite("hit_and_miss_lena.bmp", fimg) #write the output of image

## one thing need to consider, in this version do not support add and substraction since the value of kernel is always 0

def main():
    
    img = load_lena()
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]
  

    closeOp(img)
    openOp(img)
    dilation(img,kernel,5,5,2,2)
    erosion(img,kernel,5,5,2,2)
    
if __name__ == '__main__':
    main()