import cPickle
import numpy as np
import cv2
import theano
from NeuralNetTheano import *

def load_pickle(netFile):
    f = open(netFile+'.pkl',"rb")
    net = cPickle.load(f)
    f.close()
    return net

def RunVideo(filename = r"C:/Users/Akash/Desktop/DataSet/DATA/four.mp4",netFile ="stored_weights"):
    
    ##print weights , biases ,sizes
    net = load_pickle(netFile)
    weights = net.weights
    biases = net.biases
    sizes = net.sizes
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
        k = cv2.waitKey(1)
        if(k ==ord("q") & 0xff):
            break
    cap.release()
    cv2.destroyAllWindows()
