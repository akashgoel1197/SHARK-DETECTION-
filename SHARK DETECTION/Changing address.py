import os
import sys
import cv2
#To change the name of the Images and storing them in another folder
file_type = "neg"
store_folder = "neg1"
num =1430
for image in os.listdir(file_type):
    current_image =  str(file_type)+'/'+str(image)
    current = cv2.imread(current_image)
    if not os.path.exists(store_folder):
        os.makedirs(store_folder)
    cv2.imwrite(store_folder+'/'+str(num)+".jpg",current)
    num+=1
