import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

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
    return binimg

def lena_histogram():
    img = cv2.imread('lena.bmp') #read image
    # get image information
    (imgHeight, imgWidth) = img.shape[:2]

    print("This is the histogram of lena.bmp")

    histoy = np.zeros(256)
    histox = np.zeros(256)
    histox = histox.astype(int)
    histoy = histoy.astype(int)

    for i in range(imgHeight): # height 512
        for j in range(imgWidth): # width 512
            histoy[int(img[i][j].mean())] +=1
    for i in range(256):
        histox[i] = i
    plt.plot(histox, histoy) # for pixel value form 0-255
    plt.title("Lena histogram")
    plt.ylabel("# of pixels") # y label
    plt.xlabel("pixel value") # x label
    plt.savefig("lena_histogram")

def find_component_lena(img):

    (imgHeight, imgWidth) = img.shape[:2]
    
    img_map = np.zeros((imgWidth,imgHeight))
    # turn img to simple 512*512 array
    for i in range(imgWidth):
        for j in range(imgHeight):
            img_map[i][j] = int(img[i][j].mean())

    img_map = img_map.astype(int)
    # initial all the pixel with unique label
    count = int(1)
    for i in range(imgWidth):
        for j in range(imgHeight):
            if img_map[i][j] == 255: # if the pixel is 1
                img_map[i][j] = count
                count +=1

    # iterative
    change = True
    while change is True:
        change = False
        # top - down        
        for i in range(imgWidth):
            for j in range(imgHeight):
                if i-1 >=0:
                    if img_map[i][j] > img_map[i-1][j] and img_map[i-1][j] !=0:
                        img_map[i][j] = img_map[i-1][j]
                        change = True
                if j-1 >=0:
                    if img_map[i][j] > img_map[i][j-1] and img_map[i][j-1] !=0:        
                        img_map[i][j] = img_map[i][j-1]
                        change = True
        
        # down - top
        for i in range(imgWidth):
            for j in range(imgHeight):
                if imgWidth-1-i+1  < imgWidth:
                    if img_map[imgWidth-1-i][imgHeight-1-j] > img_map[imgWidth-1-i+1][imgHeight-1-j] and img_map[imgWidth-1-i+1][imgHeight-1-j] !=0:
                        img_map[imgWidth-1-i][imgHeight-1-j] = img_map[imgWidth-1-i+1][imgHeight-1-j]
                        change = True
                if imgHeight-1-j+1 < imgHeight:
                    if img_map[imgWidth-1-i][imgHeight-1-j] > img_map[imgWidth-1-i][imgHeight-1-j+1] and img_map[imgWidth-1-i][imgHeight-1-j+1] !=0:        
                        img_map[imgWidth-1-i][imgHeight-1-j] = img_map[imgWidth-1-i][imgHeight-1-j+1]
                        change = True
        
    
    #find all the labels occur above 500
    # initializing dict to store frequency of each element
    elements_count = {}
    # iterating over the elements for frequency
    for i in range(imgHeight):
        for element in img_map[i]:
            # checking whether it is in the dict or not
            if element in elements_count:
                # incerementing the count by 1
                elements_count[element] += 1
            else:
                # setting the count to 1
                elements_count[element] = 1

    for key, value in elements_count.items():
        if value > 500 and key!= 0:
            print(f"{key}: {value}")
            #four value to represent bounding box
            count = 0
            b_top = imgHeight
            b_down = 0
            b_left = imgWidth
            b_right = 0
            s_x = 0
            s_y = 0
            for i in range(imgWidth):
                for j in range(imgHeight):
                    if img_map[i][j] == key:
                        s_y += i
                        s_x += j
                        count +=1
                        if i > b_right:
                            b_right = i
                        if i < b_left:
                            b_left = i
                        if j < b_top:
                            b_top = j
                        if j > b_down:
                            b_down = j
            # 重心位置
            s_x = s_x // count
            s_y = s_y // count
            cv2.rectangle(img, (b_top, b_left), (b_down, b_right), (255, 0, 0), 2)
            img = cv2.drawMarker(img, (s_x, s_y), (0, 0, 255), markerType=cv2.MARKER_CROSS,markerSize=20,thickness=3)
    
    cv2.imwrite("connected_components_lena.bmp", img) #write the output of image
    # cv2.imshow("output Img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    
    binimg = binarized_lena()
    lena_histogram()
    find_component_lena(binimg)

if __name__ == '__main__':
    main()