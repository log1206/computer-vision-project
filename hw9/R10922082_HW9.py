import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def load_lena():
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE) #read image
    
    # init a white canvas
    img = img.astype(np.uint8) #set the right data type
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return img

def robertsOp(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()
    f1 = [[-1,0],
            [0,1]]
    f2 = [[0,-1],
            [1,0]]

    img_pad = np.pad(img,((0, 1),(0,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            r1 = -img_pad[i][j] + img_pad[i+1][j+1]
            r2 = img_pad[i+1][j] - img_pad[i][j+1]
            g = np.sqrt((r1**2) + (r2**2))
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Roberts_lena.bmp", r_img) #write the output of image            
    return r_img


def prewittOp(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()
    f1 = [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    f2 = [[ -1, 0, 1],
          [ -1, 0, 1],
          [ -1, 0, 1]]

    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            p1 = -img_pad[i-1][j-1] -img_pad[i-1][j] - img_pad[i-1][j+1] + img_pad[i+1][j-1] + img_pad[i+1][j] + img_pad[i+1][j+1]
            p2 = -img_pad[i-1][j-1] -img_pad[i][j-1] - img_pad[i+1][j-1] + img_pad[i-1][j+1] + img_pad[i][j+1] + img_pad[i+1][j+1]
            g = np.sqrt((p1**2) + (p2**2))
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Prewitt_lena.bmp", r_img) #write the output of image            
    return r_img

def sobelOp(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()
    f1 = [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]]
    f2 = [[ -1, 0, 1],
          [ -2, 0, 2],
          [ -1, 0, 1]]

    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            p1 = -img_pad[i-1][j-1] - (2*img_pad[i-1][j]) - img_pad[i-1][j+1] + img_pad[i+1][j-1] + (2*img_pad[i+1][j]) + img_pad[i+1][j+1]
            p2 = -img_pad[i-1][j-1] - (2*img_pad[i][j-1]) - img_pad[i+1][j-1] + img_pad[i-1][j+1] + (2*img_pad[i][j+1]) + img_pad[i+1][j+1]
            g = np.sqrt((p1**2) + (p2**2))
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Sobel_lena.bmp", r_img) #write the output of image            
    return r_img

def FreiAndChenOp(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()
    f1 = [[-1,-np.sqrt(2),-1],
          [ 0, 0, 0],
          [ 1, np.sqrt(2), 1]]
    f2 = [[ -1, 0, 1],
          [ -np.sqrt(2), 0, np.sqrt(2)],
          [ -1, 0, 1]]

    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            p1 = -img_pad[i-1][j-1] - (np.sqrt(2)*img_pad[i-1][j]) - img_pad[i-1][j+1] + img_pad[i+1][j-1] + (np.sqrt(2)*img_pad[i+1][j]) + img_pad[i+1][j+1]
            p2 = -img_pad[i-1][j-1] - (np.sqrt(2)*img_pad[i][j-1]) - img_pad[i+1][j-1] + img_pad[i-1][j+1] + (np.sqrt(2)*img_pad[i][j+1]) + img_pad[i+1][j+1]
            g = np.sqrt((p1**2) + (p2**2))
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Frei_and_Chen_lena.bmp", r_img) #write the output of image            
    return r_img

def kirschCompass(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()

    f1 = np.array([[[-3,-3,5],
          [ -3, 0, 5],
          [ -3, -3, 5]],
            [[-3, 5, 5],
          [ -3, 0, 5],
          [ -3, -3, -3]],
            [[5,5,5],
          [ -3, 0, -3],
          [ -3, -3, -3]],
            [[5,5,-3],
          [ 5, 0, -3],
          [ -3, -3, -3]],
            [[5,-3,-3],
          [ 5, 0, -3],
          [ 5, -3, -3]],
            [[-3,-3,-3],
          [ 5, 0, -3],
          [ 5, 5, -3]],
            [[-3,-3,-3],
          [ -3, 0, -3],
          [ 5, 5, 5]],
            [[-3,-3,-3],
          [ -3, 0, 5],
          [ -3, 5, 5]]])
  

    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            g =0
            sm = img_pad[i:i+3][:]
            sm = sm.T[j:j+3][:]
            sm =sm.T
            
            for k in range(8):
                temp = sm * f1[k]
                temp = temp.sum()
                if g < temp:
                    g = temp
            
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Kirsch_compass_lena.bmp", r_img) #write the output of image            
    return r_img

def robinsonCompass(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()

    f1 = np.array([[[-1,0,1],
          [ -2, 0, 2],
          [ -1, 0, 1]],
            [[0, 1, 2],
          [ -1, 0, 1],
          [ -2, -1, 0]],
            [[1,2,1],
          [ 0, 0, 0],
          [ -1, -2, -1]],
            [[2,1,0],
          [ 1, 0, -1],
          [ 0, -1, -2]],
            [[1,0,-1],
          [ 2, 0, -2],
          [ 1, 0, -1]],
            [[0,-1,-2],
          [ 1, 0, -1],
          [ 2, 1, 0]],
            [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]],
            [[-2,-1,0],
          [ -1, 0, 1],
          [ 0, 1, 2]]])
  

    img_pad = np.pad(img,((1, 1),(1,1)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            g =0
            sm = img_pad[i:i+3][:]
            sm = sm.T[j:j+3][:]
            sm =sm.T
            
            for k in range(8):
                temp = sm * f1[k]
                temp = temp.sum()
                if g < temp:
                    g = temp
            
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Robinson_compass_lena.bmp", r_img) #write the output of image            
    return r_img

def nevatiaBabu(img, th):
    (imgHeight, imgWidth) = img.shape[:2]
    r_img = img.copy()

    f1 = np.array([[[100,100,100,100,100],
                    [100,100,100,100,100],
                    [  0,  0,  0,  0,  0],
                    [-100,-100,-100,-100,-100],
                    [-100,-100,-100,-100,-100]],
                   [[100,100,100,100,100],
                    [100,100,100, 78,-32],
                    [100, 92, 0,-92,-100],
                    [ 32,-78,-100,-100,-100],
                    [-100,-100,-100,-100,-100]],
                   [[100,100,100, 32,-100],
                    [100,100, 92,-78,-100],
                    [100,100,  0,-100,-100],
                    [100, 78,-92,-100,-100],
                    [100,-32,-100,-100,-100]],
                   [[-100,-100,  0,100,100],
                    [-100,-100,  0,100,100],
                    [-100,-100,  0,100,100],
                    [-100,-100,  0,100,100],
                    [-100,-100,  0,100,100]],
                   [[-100, 32,100,100,100],
                    [-100, -78, 92,100,100],
                    [-100,-100,  0,100,100],
                    [-100,-100,-92, 78,100],
                    [-100,-100,-100,-32,100]],
                   [[100,100,100,100,100],
                    [-32, 78,100,100,100],
                    [-100,-92,  0, 92,100],
                    [-100,-100,-100,-78,32],
                    [-100,-100,-100,-100,-100]]])
  

    img_pad = np.pad(img,((2, 2),(2,2)), 'edge')
    img_pad = img_pad.astype(np.float64)

    for i in range(imgHeight):
        for j in range(imgWidth):
            g =0
            sm = img_pad[i:i+5][:]
            sm = sm.T[j:j+5][:]
            sm =sm.T
            
            for k in range(6):
                temp = sm * f1[k]
                temp = temp.sum()
                if g < temp:
                    g = temp
            
            if g >th:
                r_img[i][j] = 0
            else:
                r_img[i][j] = 255

    r_img = r_img.astype(np.uint8)
    cv2.imwrite("Nevatia_Babu_lena.bmp", r_img) #write the output of image            
    return r_img


def main():
    
    img = load_lena()
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]

    robertsOp(img, 12)
    prewittOp(img, 24)
    sobelOp(img, 38)
    FreiAndChenOp(img, 30)
    kirschCompass(img, 135)
    robinsonCompass(img, 43)
    nevatiaBabu(img, 12500)

    
if __name__ == '__main__':
    main()