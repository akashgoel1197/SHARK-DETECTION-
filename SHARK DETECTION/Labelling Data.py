import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# Make a pickle file containg all the images with appropiate labels
def Label(folders = ["Data/pos_background","Data/neg","Data/Dolphine"],file_store="background_removed_2",positive="Data/pos_background",shape=(30000,1)):
    """Store Data in a pickle file such that the file contain
        List of tuple
        where each tuple contain 2 elements:
        first element is the image itself
        second value is integer representing it's shark or not
    """
    x =[]
    y =[] 
    for folder in folders:
        for image in os.listdir(folder):
            current_image = folder+'/'+image
            current = mpimg.imread(current_image)

            yt=0
            if(folder==positive):
                yt=1
            
            re_image = np.reshape(current,shape)
            x.append(re_image)
            y.append(yt)

    Data = zip(x,y)
    random.shuffle(Data)
    out = open(file_store+".pkl", "wb")
    pickle.dump(Data, out)
    out.close()
