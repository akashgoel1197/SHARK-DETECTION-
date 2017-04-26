import cv2
import numpy as np
import os
import urllib2
import urllib
# TO download images form ImageNet
def store_raw_images(link,folder):
    image_urls = urllib2.urlopen(link).read()
    pic_num = 1
    if not os.path.exists(folder):
        os.makedirs(folder)
        pic_num = 1
    else:
        pic_num = len(os.listdir(folder))
    image_urls = image_urls.split('\n')
    for i in image_urls:
        try:
            print(i)
            urllib.urlretrieve(i,folder+"/"+str(pic_num)+".jpg")
            img = cv2.imread(folder+"/"+str(pic_num)+".jpg")
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite(folder+"/"+str(pic_num)+".jpg",resized_image)
            pic_num+=1
        except Exception as e:
            print (str(e))


def find_uglies(folders):
    #folders are list of folder here
    for file_type in folders:
        for image in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image = str(file_type)+'/'+str(image)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    current = cv2.imread(current_image)
                    if ugly.shape==current.shape and not(np.bitwise_xor(ugly,current).any()):
                        print('Ahmm Ugly! Delete It')
                        print(current_image)
                        os.remove(current_image)
                except Exception as e:
                    print str(e)

find_uglies(["Dolphine"])
