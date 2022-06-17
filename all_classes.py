import numpy as np #Importing all libraries
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import misc,ndimage
import pandas as pd
import os
import sys
import math
import seaborn as sns
from PIL import Image
import urllib.request

#class1
class reading_image(): #reading image from URL
        
    def reading(self):
        URL = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

        with urllib.request.urlopen(URL) as url:
            with open('temp2.jpg', 'wb') as f:
                f.write(url.read())

        self.img = cv.imread('temp2.jpg')
        return self.img
    
    def graying(self):
        gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        return gray

#class2

class addFilter():
    
    def __init__(self, image, magnitude):
        self.image = image
        self.magnitude = magnitude
#add noise to the image    
    def addNoise(self):
        temp=[]
        for x in np.nditer(self.image):
            random=np.random.rand()
            if (self.magnitude)>random:
                x=np.random.randint(255) # If magnitude is greater than random numeber then add noise to that pixel
            temp.append(x)
        temp=np.array(temp,dtype=np.uint8).reshape(self.image.shape[0],self.image.shape[1],self.image.shape[2])  
        return temp

  #Brighthen image
    def Brighten(self):
 
        temp=[]
        for x in np.nditer(self.image):
            x=x*self.magnitude
            if x>255:
                x=255
            temp.append(x)
        temp=np.array(temp,dtype=np.uint8).reshape(self.image.shape[0],self.image.shape[1],self.image.shape[2])  
        return temp

#scaling the image
    def Upscaling(self):
        
        image_upscaled = np.zeros((self.image.shape[0] * self.magnitude, self.image.shape[1] * self.magnitude,self.image.shape[2]))
        i,j=0,0
        for x in range(self.image.shape[0]):
            j=0
            for y in range(self.image.shape[1]):
                for s in range(self.magnitude):
                    image_upscaled[i+s,j+s,:]=(self.image[x,y,:])
                j+=(self.magnitude)
            i+=(self.magnitude)
            image_upscaled=np.array(image_upscaled,dtype=np.uint8)
        return image_upscaled
        
    def Downsample(self):
        image_downscaled = np.zeros((int(self.image.shape[0] * self.magnitude),int(self.image.shape[1]*self.magnitude),self.image.shape[2])) 
        downscale_factor=int(1/self.magnitude)
        for x in range(image_downscaled.shape[0]):
            for y in range(image_downscaled.shape[1]):
                 image_downscaled[x,y,:]=self.image[x*downscale_factor,y*downscale_factor,:]
        image_downscaled=np.array(image_downscaled,dtype=np.uint8)
        return image_downscaled
#class3

class Blurring():
    
    def __init__(self,img):
        self.img = img
    
    def Gaussian(self):
        print("             Gaussian Filter Blurring")
        blur = cv.GaussianBlur(self.img,(7,7),1)
        plt.imshow(blur)
        plt.subplot(121),plt.imshow(self.img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(blur),plt.title('Gaussian')
        plt.xticks([]), plt.yticks([])
        plt.show()
   
    def Median(self):
        print("Median Filter Blurring")
        median = cv.medianBlur(self.img,19)
        plt.subplot(121),plt.imshow(self.img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(median),plt.title('Median')
        plt.xticks([]), plt.yticks([])
        plt.show()


 #using oop to implement diffrent edge detection on image

#class4
class edge_detect():

    img = cv.imread('temp2.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    def __init__(self,name=gray):
        self.name=name

    def original_image(self):
        plt.imshow(self.name,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    def sobelx(self):
        sobelx1 = cv.Sobel(self.name,cv.CV_64F,1,0,ksize=5) # x-axis
        sobely1 = cv.Sobel(self.name,cv.CV_64F,0,1,ksize=5) # y-axis
        combined_sobel=cv.bitwise_or(sobelx1,sobely1)
        plt.imshow(sobelx1),plt.title('sobel X-Axis')
        plt.show()
        
    def sobely(self):
        sobelx1 = cv.Sobel(self.name,cv.CV_64F,1,0,ksize=5) # x-axis
        sobely1 = cv.Sobel(self.name,cv.CV_64F,0,1,ksize=5) # y-axis
        combined_sobel=cv.bitwise_or(sobelx1,sobely1)
        plt.imshow(sobely1),plt.title('Sobel Y-Axis')
        plt.show()
        
    def sobelcombine(self):
        sobelx1 = cv.Sobel(self.name,cv.CV_64F,1,0,ksize=5) # x-axis
        sobely1 = cv.Sobel(self.name,cv.CV_64F,0,1,ksize=5) # y-axis
        combined_sobel=cv.bitwise_or(sobelx1,sobely1)
       # plt.figure(figsize=[100,100])
        plt.imshow(combined_sobel),plt.title('combined_sobel')
        plt.show()
    
    def laplacian(self):
        laplacian1 = cv.Laplacian(self.name, cv.CV_64F)
        laplacian1=np.uint8(np.absolute(laplacian1))
        plt.imshow(laplacian1),plt.title('Laplacian Edge Detection View')
        plt.show()
        
    def canny(self):
        canny = cv.Canny(self.name,50,200)
        plt.imshow(canny,cmap = 'gray'),plt.title('Canny Edge Detection View')

    ##HoughLines


    def HoughLines():
        
        #img = cv.imread('temp2.jpg')
        default_file = 'temp2.jpg'
        filename = default_file

        # Loads an image
        src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

        dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
    
        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
        cv.waitKey()
        return 0