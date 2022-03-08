import numpy as np
import cv2
import matplotlib.pyplot as plt #plot histogram

def load_img(name):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE) #read image
    # init a white canvas
    img = img.astype(np.uint8) #set the right data type
    # cv2.imwrite("binarized_lena.bmp", binimg) #write the output of image
    return img

def mypsnr(nsimg):
    
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE) #read image
    img = img.astype(np.uint8) #set the right data type
    img = img.astype(np.float64)
    nsimg = nsimg.astype(np.float64)

    #normalize ??
    # img *= 255.0/img.max()
    # nimg *= 255.0/nimg.max()

    #calculate std and SNR
    VS = np.std(img)
    mu = np.mean(img) #not use
    VN = np.std(nsimg-img)
    mun = np.mean(nsimg-img) #not use
    SNR = 20 * np.log10(VS/ VN)
    return SNR

def dilation(img, kernel, k_h, k_w, centerX, centerY):      # align center to the pixel, add together and pick the max value
    # print("do dilation")
    (imgHeight, imgWidth) = img.shape[:2]
    
    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight))
    for i in range(imgHeight):
        for j in range(imgWidth):
            max = 0#np.array([0,0,0]) # temp value for every pixel in order to find max value under kernel
            for a in range(k_h):
                for b in range(k_w):
                    if kernel[a][b] ==1 and i+a-centerY < imgHeight and j+b-centerX < imgWidth and i+a-centerY >=0 and j+b-centerX >=0 :
                        if img[i+a-centerY][j+b-centerX] > max:
                            max = img[i+a-centerY][j+b-centerX]
            nimg[i][j] =  max

    nimg = nimg.astype(np.uint8) #set the right data type
    # cv2.imwrite("dilation_lena.bmp", nimg) #write the output of image

    
    # cv2.imshow("output Img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return nimg

def erosion(img,kernel, k_h,k_w, centerX,centerY):       # align center to the pixel, substract by kernel and pick minimum
    # print("do erosion")
    (imgHeight, imgWidth) = img.shape[:2]
    # represent kernel information center point and half width and height

    # init a white canvas
    nimg = np.zeros(shape=(imgWidth,imgHeight))
    for i in range(imgHeight):
        for j in range(imgWidth):
            min = 255#np.array([255,255,255]) 
            for a in range(k_h):  # kernel operation
                for b in range(k_w):
                    if kernel[a][b] ==1 and i+a-centerY < imgHeight and j+b-centerX < imgWidth and i+a-centerY >=0 and j+b-centerX >=0 :
                        if img[i+a-centerY][j+b-centerX] < min:
                            min = img[i+a-centerY][j+b-centerX]
            nimg[i][j] =  min 

    
    nimg = nimg.astype(np.uint8) #set the right data type
    # cv2.imwrite("erosion_lena.bmp", nimg) #write the output of image

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
    # cv2.imwrite("closing_lena.bmp", c_img) #write the output of image
      
    # cv2.imshow("output Img c", c_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return c_img

def openOp(img):        # erosion then dilation
    # print("do open")
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]
    e_img = erosion(img,kernel,5,5,2,2)
    o_img = dilation(e_img,kernel,5,5,2,2)
    # cv2.imwrite("opening_lena.bmp", o_img) #write the output of image

    # cv2.imshow("output Img o", o_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return o_img


def gaussian_noise(img, amp):
    f_img = img.copy()
    (imgHeight, imgWidth) = img.shape[:2]
    f_img = f_img.astype(np.float64)
    f_img = f_img + amp*np.random.normal(0, 1, size=(imgHeight, imgWidth))
    for i in range(imgHeight): # check if over 255
        for j in range(imgWidth):
            if f_img[i][j] > 255:
                f_img[i][j] = 255
    f_img = f_img.astype(np.uint8)
    cv2.imwrite("lena_gaussian_"+ str(amp) +".bmp", f_img) #write the output of image
    print("lena_gaussian_"+ str(amp) +": ", mypsnr(f_img))
    return f_img

def salt_and_pepper(img, th):
    f_img = img.copy()
    (imgHeight, imgWidth) = img.shape[:2]
    noise = np.random.uniform(low=0.0, high=1.0, size=(imgHeight, imgWidth))
    for i in range(imgHeight): # check if over 255
        for j in range(imgWidth):
            if noise[i][j] < th:
                f_img[i][j] = 0
            elif noise[i][j] > (1-th):
                f_img[i][j] = 255
    f_img = f_img.astype(np.uint8)
    cv2.imwrite("lena_salt_and_pepper_"+ str(th) +".bmp", f_img) #write the output of image
    snr = mypsnr(f_img)
    print("lena_salt_and_pepper_"+ str(th) +": ", snr)
    return f_img

def box_filter(img, sz, noisename):
    f_img = img.copy()
    (imgHeight, imgWidth) = img.shape[:2]
    #padding bt edge value
    sz_l = (sz-1)//2
    img_pad = np.pad(img,((sz_l, sz_l),(sz_l,sz_l)), 'edge')
    for i in range(imgHeight):
        for j in range(imgWidth):
            re = 0
            for a in range(sz):
                for b in range(sz):
                    re += int(img_pad[i+a][j+b])
            f_img[i][j] = re//(sz**2)

    f_img = f_img.astype(np.uint8)
    cv2.imwrite("lena_box_"+ str(sz) +"x"+ str(sz) +"_"+noisename+".bmp", f_img) #write the output of image
    print("lena_box_"+ str(sz) +"x"+ str(sz) +"_"+noisename+": ", mypsnr(f_img))

def median_filter(img, sz, noisename):
    f_img = img.copy()
    (imgHeight, imgWidth) = img.shape[:2]
    #padding bt edge value
    sz_l = (sz-1)//2
    img_pad = np.pad(img,((sz_l, sz_l),(sz_l,sz_l)), 'edge')
    for i in range(imgHeight):
        for j in range(imgWidth):
            # sm = img_pad[i-sz_l : i-sz_l+sz][j-sz_l : j-sz_l+sz]
            sm = img_pad[i : i+sz][:]
            sm = sm.T[j : j+sz][:]
            # print("sm: ",sm)
            med = np.median(sm)
            # print("med: ",med)
            f_img[i][j] = med

    f_img = f_img.astype(np.uint8)
    cv2.imwrite("lena_median_"+ str(sz) +"x"+ str(sz) +"_"+noisename+".bmp", f_img) #write the output of image
    print("lena_median_"+ str(sz) +"x"+ str(sz) +"_"+noisename+": ", mypsnr(f_img))

def o_t_c(img, noisename):
    o_img = openOp(img)
    c_img = closeOp(o_img)
    cv2.imwrite("lena_open_then_close_"+noisename+".bmp", c_img) #write the output of image
    print("lena_open_then_close_"+noisename+": ", mypsnr(c_img))


def c_t_o(img, noisename):
    c_img = closeOp(img)    
    o_img = openOp(c_img)
    cv2.imwrite("lena_close_then_open_"+noisename+".bmp", o_img) #write the output of image
    print("lena_close_then_open_"+noisename+": ", mypsnr(o_img))


def all_filter(img, nn):

    box_filter(img, 3, nn)
    box_filter(img, 5, nn)
    median_filter(img, 3, nn)
    median_filter(img, 5, nn)
    o_t_c(img, nn)
    c_t_o(img, nn)



def main():
    
    img = load_img('lena.bmp')
    # nimg = load_img('median_5x5.bmp')
    kernel =[[0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]

    img1 = gaussian_noise(img,10)
    img2 = gaussian_noise(img,30)
    img3 = salt_and_pepper(img,0.05)
    img4 = salt_and_pepper(img,0.1)

    all_filter(img1, "gaussian_10")
    all_filter(img2, "gaussian_30")
    all_filter(img3, "salt_and_pepper_0.05")
    all_filter(img4, "salt_and_pepper_0.1")
    

if __name__ == '__main__':
    main()


# ii = kernel[:][0:2]
# print(ii)

# kernel1 =np.array([[0,1,1,1,0],
#         [1,1,1,1,1],
#         [1,1,3,7,1],
#         [1,1,5,2,1],
#         [0,1,1,8,0]])
# ii = kernel1[2:5][:]
# ii = ii.T[0:3][:]
# ii =ii.T
# print(ii)
# ie = np.median(ii)
# print(ie)
# print(n_img)
# cv2.imshow("noise 10", n_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()