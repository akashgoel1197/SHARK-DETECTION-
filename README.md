# SHARK-DETECTION-
In this project i have tried to develop a neural network to detect shark in videos and images. This project is in python using openCV.
The accuracy of the model is around 65% and the model is trained on a dataset of 10,000 images in which 5000 images were of shark and 5000 images were of non-shark.The images were handcrafted and downloaded using urllib and beautifulsoup.
files discryption:

PythonScript.py : Python file to download images from the web using urllib2.
background.py      : This file remove the background from an image using Canny edge detection and contours.
LabelingData.py : This file automatically label the data given in the folder.
ConvertToGrayImage : This file convert the images to gray images and make a pickle file consisting of all the images.
NeuralNetBasic.py : Implementation of Neural Network.
load_data.py      : Load's the dataset and convert into a suitable form to be given as an input to the input layer of Neural Network.
RunonVideo        : give the result of detection in the user's video.
