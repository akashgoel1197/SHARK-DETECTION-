# SHARK-DETECTION-
In this project i have tried to develop a neural network to detect shark in videos and images. This project is in python using openCV.
The accuracy of the model is around 65% and the model is trained on a dataset of 10,000 images in which 5000 images were of shark and 5000 images were of non-shark.The images were handcrafted and downloaded using urllib and beautifulsoup.


1. PythonScript.py : Python file to download images from the web using urllib2.
2. background.py      : This file remove the background from an image using Canny edge detection and contours.
3. LabelingData.py : This file automatically label the data given in the folder.
4. ConvertToGrayImage : This file convert the images to gray images and make a pickle file consisting of all the images.
5. NeuralNetBasic.py : Implementation of Neural Network.
6. load_data.py      : Load's the dataset and convert into a suitable form to be given as an input to the input layer of Neural Network.
7. RunonVideo        : give the result of detection in the user's video.
