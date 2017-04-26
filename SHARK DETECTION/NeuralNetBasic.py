import gzip 
import cPickle
import numpy as np
import random
import time
import matplotlib.pyplot as plt


    
class NeuralNet(object):
    def __init__(self, sizes,weights=None,biases=None):
        self.sizes = sizes
        self.num_layers = len(sizes)
        if(weights is None and biases is None):
             self.biases = [np.random.randn(x,1) for x in sizes[1:]]
             self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        else:
            self.biases = biases
            self.weights = weights
        self.data=[]
    
    def feedforward(self, inp):
        """
        Returns the output of a feedfoward network wen input inp is given
        """
        
        a = inp
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b) 
        return a
        
    def SGD(self, td, epochs, mbs, eta, test_data = None):
        """
        Stochastic Gradient Descent.
        
        td:         training data to perform SGD upon.
        epochs:     Number of epochs or full iterations over the dataset.
        mbs:        Size of mini-batch used.
        eta:        Learning Rate
        test_data:  If test data is present, the function tests the model over
                    test data, and returns the accuracy.
        
        """
        for x in xrange(epochs):
            
            mini_batches = []
            random.shuffle(td)
            
            for i in range(0, len(td), mbs):
                mini_batches.append(np.array(td[i:i+mbs]))
                
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch :",x, "Accuracy:", self.evaluate(test_data), "/", len(test_data))
            else:
                res = [(np.argmax(self.feedforward(a)),np.argmax(b))
                       for (a,b) in td]
                acc = sum(int(a==b) for (a,b) in res)
                
                print("Epoch :",x, "Accuracy:", acc, "/", len(td))
                self.data.append(acc)
##                plt.plot(np.arange(x),data)
                print("Epoch :",x,"Completed")
        
                
    def update_mini_batch(self, mini_batches, eta):
        
        """
        Updates the parameters of the model using the backpropogation algorithm
        over all examples.
        
        mini_batches: array of mini_batches
        eta         : Learning Rate
        """
        
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        xv = np.asarray([x.ravel() for (x,y) in mini_batches]).transpose()
        yv = np.asarray([y.ravel() for (x,y) in mini_batches]).transpose()
        
        delta_b, delta_w = self.backprop(xv,yv)
        
        nabla_w = [nw + ndw for  nw, ndw in zip(nabla_w, delta_w)]
        nabla_b = [nb + ndb for  nb, ndb in zip(nabla_b, delta_b)]
        self.weights = [w-(eta/len(mini_batches))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b-(eta/len(mini_batches))*nb for b, nb in zip(self.biases, nabla_b)]
            
    def backprop(self,x,y):
        """
        Backpropogation Algorithm. Calculates the gradient for the entire set
        of paramters of a model given a training example and it's output using
        Backpropogation.
        """
        
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
    
        delta = (self.cost_derivative(activations[-1], y))*(sigmoid_prime(zs[-1]))
        delta_s = delta.sum(1).reshape(len(delta), 1)
    
        nabla_b[-1] = delta_s
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for i in xrange(2, self.num_layers):
            
            delta = (np.dot(self.weights[-i + 1].transpose(), delta))*sigmoid_prime(zs[-i])
            delta_s = delta.sum(1).reshape(len(delta), 1)

            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
            nabla_b[-i] = delta_s
        
        return nabla_b, nabla_w

            
    def evaluate(self, test_data):
        """
        Evaluates the performance of the neural over test data.
        Returns classification accuracy.
        
        """
        
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]       
        return sum([int(x==y) for x,y in test_results])
    
    def cost_derivative(self, output_activations, y):
           """Return the vector of partial derivatives \partial C_x /
           partial a for the output activations."""
           return (output_activations-y)
           
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
     



    
