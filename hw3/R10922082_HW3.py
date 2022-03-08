import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def lena_histogram(path,name):
    img = cv2.imread(path) #read image
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]

    histoy = np.zeros(256)
    histox = np.zeros(256)
    histox = histox.astype(int)
    histoy = histoy.astype(int)

    plt.clf()

    for i in range(imgHeight): # height 512
        for j in range(imgWidth): # width 512
            histoy[int(img[i][j].mean())] +=1
    for i in range(256):
        histox[i] = i
    plt.plot(histox, histoy) # for pixel value form 0-255
    plt.title("Lena histogram " + name)
    plt.ylabel("# of pixels") # y label
    plt.xlabel("pixel value") # x label
    plt.savefig("lena_histogram_" + name)

    return histoy

def ope_image(rate):
    img = cv2.imread('lena.bmp') #read image
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight,3))
    # manipulate pixels
    for i in range(imgHeight):
        for j in range(imgWidth):
            
            nimg[i][j]= img[i][j]/rate

    nimg = nimg.astype(np.uint8) #set the right data type
    cv2.imwrite("divide"+ str(rate) +"_lena.bmp", nimg) #write the output of image


    # cv2.imshow("output Img", nimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def equal_image(histo, path):


    # histo is 1*256
    img = cv2.imread(path) #read image
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]

    N = imgHeight * imgWidth # # of pixels
    table = np.zeros(256)
    table = table.astype(int)
    for i in range(len(histo)):
        table[i] = histo[i]*255 / N # probability of each value

    for i in range(len(histo)):
        if i > 0:
            table[i] = table[i] + table[i-1]
        #print(table[i])
        table[i] = round(table[i])

    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight,3))

    for i in range(imgHeight): # height 512
        for j in range(imgWidth): # width 512
            nimg[i][j] = table[int(img[i][j].mean())]

    nimg = nimg.astype(np.uint8) #set the right data type
    cv2.imwrite("equalized_lena.bmp", nimg) #write the output of image

    # cv2.imshow("output Img", nimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def main():
    
  
    histo = lena_histogram('lena.bmp',"origin")
    ope_image(3)
    histo = lena_histogram('divide3_lena.bmp',"divide_by_3")
    equal_image(histo, 'divide3_lena.bmp')
    lena_histogram('equalized_lena.bmp',"equalized")
    
if __name__ == '__main__':
    main()