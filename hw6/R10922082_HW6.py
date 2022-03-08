from os import write
from types import resolve_bases
import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def load_lena():
    img = cv2.imread('lena.bmp') #read image
    
    # init a white canvas
    img = img.astype(np.uint8) #set the right data type
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return img


def binarized_lena(img):

    
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
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return binimg

def down_sample(img):
    # take every top-left one to represent 8 * 8 original is 512 * 512
    # get image information 
    (imgHeight, imgWidth) = img.shape[:2]
    bs = 8 # block size
    # init a white canvas
    dimg = np.zeros(shape=(imgWidth//bs,imgHeight//bs,3))
    (imgHeight, imgWidth) = dimg.shape[:2] # new width and height
    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            
            dimg[i][j]= img[i*bs][j*bs]         

    dimg = dimg.astype(np.uint8) #set the right data type
    # cv2.imwrite("down_sample_lena.bmp", dimg) #write the output of image
    # print(dimg.shape)
    return dimg


def yokoiCnum(img):

    # image to label matrix
    global labelm
    (imgHeight, imgWidth) = img.shape[:2]
    labelm = np.zeros((imgWidth +2,imgHeight +2)) # for label with padding
    yokoim = np.zeros((imgWidth ,imgHeight)) # for result
    for i in range(imgHeight):
        for j in range(imgWidth):
            if img[i][j].mean() == 0:
                labelm[i+1][j+1] = 0
            else:
                labelm[i+1][j+1] = 1
    labelm = labelm.astype(np.int)
    # do algorithm
    for i in range(imgHeight):
        for j in range(imgWidth):
            yokoim[i][j] = f(i+1,j+1) #map to label matrix position

    yokoim = yokoim.astype(np.int)
    ## write the output matrix to txt file
    with open("lena_yokoi_connectivity_number.txt", mode="w", encoding="utf-8") as file:
        for i in range(yokoim.shape[0]):
            for j in range(yokoim.shape[1]):
                if yokoim[i][j] == 0:
                    file.write(" ")
                else:
                    file.write(str(yokoim[i][j]))
            file.write("\n")

# let q = 1  r = 2 s =0
def isR(a): # check if a is R
    if a == 2:
        return True
    else:
        return False 

def isQ(a): # check if a is Q
    if a == 1:
        return 1
    else:
        return 0

def index2pos(index): ## decode index to x,y position
    if index == 1:
        return (1,0)
    elif index == 2:
        return (0,-1)
    elif index == 3:
        return (-1,0)
    elif index == 4:
        return (0,1)
    elif index == 5:
        return (1,1)
    elif index == 6:
        return (1,-1)
    elif index == 7:
        return (-1,-1)
    elif index == 8:
        return (-1,1)
    else:
        return (0,0)
    
def h(y,x,c,d,e):
    global labelm
    (cx,cy) = index2pos(c)
    (dx,dy) = index2pos(d)
    (ex,ey) = index2pos(e)
    
    if labelm[y][x] == 1: # center exist 
        # checl b != c
        if labelm[y][x] != labelm[y+cy][x+cx]:
            return 0 #s
        else:
            if labelm[y][x] ==labelm[y+ey][x+ex] and labelm[y+dy][x+dx] ==labelm[y][x]:
                return 2 #r
            else:
                return 1 #q

def f(y,x):

    a1 = h(y,x,1,6,2)
    a2 = h(y,x,2,7,3)
    a3 = h(y,x,3,8,4)
    a4 = h(y,x,4,5,1)

    if isR(a1) and isR(a2) and isR(a3) and isR(a4):
        return 5 # internal
    else:
        return isQ(a1) + isQ(a2) + isQ(a3) + isQ(a4)
    

def main():
    
    img = load_lena()
    # kernel =[[0,1,1,1,0],
    #         [1,1,1,1,1],
    #         [1,1,1,1,1],
    #         [1,1,1,1,1],
    #         [0,1,1,1,0]]
  
    bimg = binarized_lena(img)
    
    dimg = down_sample(bimg)

    yokoiCnum(dimg)
    # for i in range(9):
    #     (x,y) = index2pos(i)
    #     print("(",x," ",y,")")

    
if __name__ == '__main__':
    main()