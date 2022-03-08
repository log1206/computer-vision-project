import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def load_lena():
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE) #read image
    
    # init a white canvas
    img = img.astype(np.uint8) #set the right data type
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return img



def laplacianOp1(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    l_img = img.copy()
    r_img = img.copy()
    f1 = [[ 0,  1, 0],
          [ 1, -4, 1],
          [ 0,  1, 0]]
   
    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)
    l_img = l_img.astype(np.float64)
    for i in range(imgHeight):
        for j in range(imgWidth):
            g = img_pad[i+1][j] +img_pad[i][j+1] - 4*img_pad[i+1][j+1] + img_pad[i+2][j+1] + img_pad[i+1][j+2]         
            if g >=th:
                l_img[i][j] = 1
            elif g <= -th:
               

                l_img[i][j] = -1
            else:
                l_img[i][j] = 0

    l_img_pad = np.pad(l_img,((1, 1),(1,1)), 'edge')
   
    for i in range(imgHeight):
        for j in range(imgWidth):
           
                   
            if l_img_pad[i+1][j+1] >= 1:
               
                if l_img_pad[i][j] <=-1 or l_img_pad[i+1][j] <= -1 or l_img_pad[i+2][j] <= -1 or l_img_pad[i+2][j+1] <= -1 or l_img_pad[i+2][j+2] <= -1 or l_img_pad[i+1][j+2] <= -1 or l_img_pad[i][j+2] <= -1 or l_img_pad[i][j+1] <= -1:
                    
                    r_img[i][j] = 0
                else:
                    r_img[i][j] = 255
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("laplacian1_lena.bmp", r_img) #write the output of image            
    return r_img


def laplacianOp2(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    l_img = img.copy()
    r_img = img.copy()
    f1 = [[ 1,  1, 1], ## need divide by 3
          [ 1, -8, 1],
          [ 1,  1, 1]]
   
    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)
    l_img = l_img.astype(np.float64)
    for i in range(imgHeight):
        for j in range(imgWidth):
            g = img_pad[i][j] + img_pad[i+2][j] + img_pad[i][j+2] + img_pad[i+2][j+2] + img_pad[i+1][j] +img_pad[i][j+1] - 8*img_pad[i+1][j+1] + img_pad[i+2][j+1] + img_pad[i+1][j+2]         
            g = g/3
            if g >=th:
                l_img[i][j] = 1
            elif g <= -th:
                l_img[i][j] = -1
            else:
                l_img[i][j] = 0

    l_img_pad = np.pad(l_img,((1, 1),(1,1)), 'edge')
   
    for i in range(imgHeight):
        for j in range(imgWidth):     
            if l_img_pad[i+1][j+1] >= 1:
                if l_img_pad[i][j] <=-1 or l_img_pad[i+1][j] <= -1 or l_img_pad[i+2][j] <= -1 or l_img_pad[i+2][j+1] <= -1 or l_img_pad[i+2][j+2] <= -1 or l_img_pad[i+1][j+2] <= -1 or l_img_pad[i][j+2] <= -1 or l_img_pad[i][j+1] <= -1:     
                    r_img[i][j] = 0
                else:
                    r_img[i][j] = 255
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("laplacian2_lena.bmp", r_img) #write the output of image            
    return r_img

def minVarlaplacian(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    l_img = img.copy()
    r_img = img.copy()
    f1 = [[ 2,  -1, 2], ## need divide by 3
          [ -1, -4, -1],
          [ 2,  -1, 2]]
   
    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)
    l_img = l_img.astype(np.float64)
    for i in range(imgHeight):
        for j in range(imgWidth):
            g = 2*img_pad[i][j] + 2*img_pad[i+2][j] + 2*img_pad[i][j+2] + 2*img_pad[i+2][j+2] - img_pad[i+1][j] - img_pad[i][j+1] - 4*img_pad[i+1][j+1] - img_pad[i+2][j+1] - img_pad[i+1][j+2]
            g = g/3
            if g >=th:
                l_img[i][j] = 1
            elif g <= -th:
                l_img[i][j] = -1
            else:
                l_img[i][j] = 0

    l_img_pad = np.pad(l_img,((1, 1),(1,1)), 'edge')
   
    for i in range(imgHeight):
        for j in range(imgWidth):     
            if l_img_pad[i+1][j+1] >= 1:
                if l_img_pad[i][j] <=-1 or l_img_pad[i+1][j] <= -1 or l_img_pad[i+2][j] <= -1 or l_img_pad[i+2][j+1] <= -1 or l_img_pad[i+2][j+2] <= -1 or l_img_pad[i+1][j+2] <= -1 or l_img_pad[i][j+2] <= -1 or l_img_pad[i][j+1] <= -1:     
                    r_img[i][j] = 0
                else:
                    r_img[i][j] = 255
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("minimum-variance-laplacian_lena.bmp", r_img) #write the output of image            
    return r_img


def LoG(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    l_img = img.copy()
    r_img = img.copy()
    f1 = np.array([[ 0,  0, 0, -1, -1, -2, -1, -1, 0, 0, 0], ## need divide by 3
          [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
          [ 0,  -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
          [ -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
          [ -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
          [ -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
          [ -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
          [ -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
          [ 0,  -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
          [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
          [ 0,  0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
   
    img_pad = np.pad(img,((5, 5),(5,5)), 'edge')
    img_pad = img_pad.astype(np.float64)
    l_img = l_img.astype(np.float64)
    for i in range(imgHeight):
        for j in range(imgWidth):
            sm = img_pad[i : i+11][:]
            sm = sm.T[j : j+11][:]
            sm = sm.T
            g = np.sum(sm * f1)
           
            if g >=th:
                l_img[i][j] = 1
            elif g <= -th:
                l_img[i][j] = -1
            else:
                l_img[i][j] = 0

    l_img_pad = np.pad(l_img,((1, 1),(1,1)), 'edge')
   
    for i in range(imgHeight):
        for j in range(imgWidth):     
            if l_img_pad[i+1][j+1] >= 1:
                if l_img_pad[i][j] <=-1 or l_img_pad[i+1][j] <= -1 or l_img_pad[i+2][j] <= -1 or l_img_pad[i+2][j+1] <= -1 or l_img_pad[i+2][j+2] <= -1 or l_img_pad[i+1][j+2] <= -1 or l_img_pad[i][j+2] <= -1 or l_img_pad[i][j+1] <= -1:     
                    r_img[i][j] = 0
                else:
                    r_img[i][j] = 255
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Laplacian_of_Gaussian_lena.bmp", r_img) #write the output of image            
    return r_img


def DoG(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    l_img = img.copy()
    r_img = img.copy()
    f1 = np.array([[ -1,  -3, -4, -6, -7, -8, -7, -6, -4, -3, -1], ## need divide by 3
          [ -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
          [ -4,  -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
          [ -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
          [ -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
          [ -8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
          [ -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
          [ -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
          [ -4,  -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
          [ -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
          [ -1,  -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])
   
    img_pad = np.pad(img,((5, 5),(5,5)), 'edge')
    img_pad = img_pad.astype(np.float64)
    l_img = l_img.astype(np.float64)
    for i in range(imgHeight):
        for j in range(imgWidth):
            sm = img_pad[i : i+11][:]
            sm = sm.T[j : j+11][:]
            sm = sm.T
            g = np.sum(sm * f1)
           
            if g >=th:
                l_img[i][j] = 1
            elif g <= -th:
                l_img[i][j] = -1
            else:
                l_img[i][j] = 0

    l_img_pad = np.pad(l_img,((1, 1),(1,1)), 'edge')
   
    for i in range(imgHeight):
        for j in range(imgWidth):     
            if l_img_pad[i+1][j+1] >= 1:
                if l_img_pad[i][j] <=-1 or l_img_pad[i+1][j] <= -1 or l_img_pad[i+2][j] <= -1 or l_img_pad[i+2][j+1] <= -1 or l_img_pad[i+2][j+2] <= -1 or l_img_pad[i+1][j+2] <= -1 or l_img_pad[i][j+2] <= -1 or l_img_pad[i][j+1] <= -1:     
                    r_img[i][j] = 0
                else:
                    r_img[i][j] = 255
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Difference_of_Gaussian_lena.bmp", r_img) #write the output of image            
    return r_img


def main():
    
    img = load_lena()
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]

    laplacianOp1(img,15)
    laplacianOp2(img,15)
    minVarlaplacian(img, 20)
    LoG(img, 3000)
    DoG(img, 1)

if __name__ == '__main__':
    main()