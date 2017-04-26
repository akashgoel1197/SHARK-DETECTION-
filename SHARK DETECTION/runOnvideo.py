import cPickle
import numpy as np
import cv2
from NeuralNetBasic import NeuralNet


def load_pickle(netFile):
    f = open(netFile+'.pkl',"rb")
    net = cPickle.load(f)
    f.close()
    return net
    
def RunVideo(filename = r"C:/Users/Akash/Desktop/DataSet/DATA/three.mp4",netFile ="stored_weights"):
    
    ##print weights , biases ,sizes
    net = load_pickle(netFile)
    weights = net.weights
    biases = net.biases
    sizes = net.sizes
    print sizes
    neto = NeuralNet(sizes,weights,biases)
    cap = cv2.VideoCapture(filename)

    while(True):
        ret,frame = cap.read()
        for i in range(5):
            ret,frame = cap.read()

        if(ret==False):
            break
        resize_frame = cv2.resize(frame,(100,100))
        resize_frame = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2RGB)
        resize_frame =  np.reshape(resize_frame,(30000,1))
        y_ = neto.feedforward(resize_frame)
        print(y_),np.argmax(y_)
        if(np.argmax(y_)==1):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,"Shark Detected",(frame.shape[0]/2,frame.shape[1]/2),font,3,(255,255,255),3,cv2.LINE_AA)
        cv2.imshow("img",frame)
        k = cv2.waitKey(100)
        if(k ==ord("q") & 0xff):
            break
    cap.release()
    cv2.destroyAllWindows()
def RunImage(filename = r"C:/Users/Akash/Desktop/DataSet/DATA/Test/3.jpg",netFile ="stored_weights",net=None):
##    net = load_pickle(netFile)
    weights = net.weights
    biases = net.biases
    sizes = net.sizes
    neto = NeuralNet(sizes,weights,biases)
    frame = cv2.imread(filename)
    resize_frame = cv2.resize(frame,(100,100))
    resize_frame = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2RGB)
    resize_frame =  np.reshape(resize_frame,(30000,1))
    y_ = neto.feedforward(resize_frame)
    print y_
    if(np.argmax(y_)==1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"Shark Detected",(frame.shape[0]/3,frame.shape[1]/3),font,frame.shape[0]/250.0,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imshow("img",frame)
    k = cv2.waitKey()
    cv2.destroyAllWindows()

def RunImages(filename=r"C:/Users/Akash/Desktop/DataSet/DATA/Test/",netFile="stored_weights"):
    net = load_pickle(netFile)
    for i in range(1,11):
        RunImage(filename+str(i)+'.jpg',netFile,net)
    
"""____---------------------------------------_________________-----------------------________________"""

#------------Below Commented Code is For storing The Videos as Image-----------------#
##filename = r"C:/Users/Akash/Desktop/DataSet/DATA/five.mp4"
##folder = r"C:/Users/Akash/Desktop/DataSet/Temp"
##cap = cv2.VideoCapture(filename)
##count = 0
##while(True):
##    
##    ret,frame = cap.read()
##    for i in range(5):
##        ret,frame = cap.read()
##    if(ret==False):
##        break
##    resize_frame = cv2.resize(frame,(100,100))
##    cv2.imshow("i",resize_frame)
##    print (resize_frame.shape)
##    resize_frame = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2RGB)
##    k = cv2.waitKey(0)
##    resize_frame =  np.reshape(resize_frame,(30000,1))
##    if(k == ord("q") & 0xff):
##        break
##    cv2.imwrite(folder+"/"+str(count)+".jpg",resize_frame)
##    count+=1
##
##cap.release()
##cv2.destroyAllWindows()
    
    
    
