import cPickle
import numpy as np
import random

#A file to load testing and training data
def load_data(filename = "myfile"):
    """
    Loads data from file. Returns a tuple of 3 lists, containing training data,
    validation data and test data in order. 
    x.a
    The training data , validation and test data are tuples of two numpy arrays 
    of length 10,000 each. First of these is contains 784x1 numpy arrays which 
    represents the pixel intensities of the image. The second contains integers 
    representing the correct  classification for examples of the corresponding
    indexes.
    """
    
    f = open(filename +'.pkl', 'rb')
    training_data =  cPickle.load(f)
    f.close()
    return training_data


def load(filename = "myfile"):
    """
    Tranform the data into a format which is more feasible for training.
    
    Returns a a 3-tuple of containing training data validation data and test
    data in order. 
    
    The training data is now an list of 50,000 tuples representing each training 
    example. Each tuple consists of a 784x1 numpy array, representing pixel 
    intensities and a 10x1 numpy array, with 0 for all indexes but 1 for theindex 
    corresponding to the correct classification of the example image.
    
    The training data is now an list of 50,000 tuples representing each training 
    example. Each tuple consists of a 784x1 numpy array, representing pixel 
    intensities and an integer  corresponding to the correct classification of
    the image example.
    """
    
    td = load_data(filename)
    te = load_data("New_Data")
    X_train = [np.reshape(x[0], (30000,1)) for x in td]
    Y_train = [vectorize(y[1]) for y in td]
    train_data = zip(X_train, Y_train)
    X_test = [np.reshape(x[0], (30000,1)) for x in te]
    Y_test= [y[1] for y in te]
    test_data = zip(X_test, Y_test)
    return train_data,test_data
    
    
    
def vectorize(s):
    """
    Returns a 10x1 numpy array with all indices 0 except for sth indice
    """
    
    result = np.zeros((2,1))
    result[s] = 1
    return result
    
