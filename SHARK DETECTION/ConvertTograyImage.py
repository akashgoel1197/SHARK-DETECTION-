
import os
import sys
#3rd Party
import matplotlib.pyplot as plt
import numpy as np
import cPickle
#Convert the image into gray scale Image and also make a pickle file
#containg all the Images.
def rgb2gray(image):
    return np.dot(image[... ,:3],[0.299,0.587,0.114])
def convertImage(image):
##    print(image)
    gray = rgb2gray(image)
##    plt.subplot(2,1,1)
##    plt.imshow(image)
##    plt.subplot(2,1,2)
##    plt.imshow(gray,cmap="gray")
##    plt.show()
    return gray

def convertAlltoGray(filename):
    f = open(filename,"rb")
    data = cPickle.load(f)
    x = [convertImage(np.reshape(point[0],(100,100,3))) for point in data]
    y = [point[1] for point in data]
    zipper = zip(x,y)
    return zipper

def makeGrayPicklefile(file_open="myfile.pkl",file_store="gray"):
    zipper = convertAlltoGray(file_open)
    out = open(file_store+".pkl","wb")
    cPickle.dump(zipper,out)
    out.close()

