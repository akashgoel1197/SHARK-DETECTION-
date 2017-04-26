import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) 

#The file Try to extract the background from all the Images and even from Video and store them in a Folder.
f = r"C:/Users/Akash/Desktop/DataSet/DATA/Dolphine"


def video_removal(video_file):
    cap = cv2.VideoCapture(video_file)
    count = 0
    while(True):
         for i in range(20):
             ret,frame = cap.read()
         if(ret==False):
             break
         frame = cv2.resize(frame,(100,100))
         masked = remove_background(frame)
         if(type(masked)=="int"):
             continue
         address = 'C:/Users/Akash/Desktop/DataSet/DATA/Video_Image'
         if not os.path.exists(address):
             os.mkdir(address)
         cv2.imwrite(address +'/'+str(count)+'.jpg', masked)# Save
         count+=1
    cv2.destroyAllWindows()
    cap.release()
def image_removal(folder =f):
    for image in os.listdir(f):
        c = f+'/'+image
        frame  = cv2.imread(c)
        masked = remove_background(frame)
        if(type(masked)=="int"):
            continue
        address = 'C:/Users/Akash/Desktop/DataSet/DATA/Dolphine_background'
        if not os.path.exists(address):
             os.mkdir(address)
        cv2.imwrite(address+'/'+image, masked)# Save
    cv2.destroyAllWindows()
         
def remove_background(frame):
    masks =[]
    
    
##    plt.subplot(4,4,1)
    for x in [0,62,127,192,255]:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,x,255,cv2.THRESH_TRUNC)
            #Detecting Edges Here
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
##        plt.subplot(4,4,2)
##        plt.imshow(edges)
        edges = cv2.dilate(edges, None)
##        plt.subplot(4,4,3)
##        plt.imshow(edges)
##        
        edges = cv2.erode(edges, None)
##        plt.subplot(4,4,4)
##        plt.imshow(edges)
##        plt.show()
    # Find contours in edges, sort by area 
        contour_info = []
        im2,contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if(len(contours)==0):
            continue
        for c in contours:
            contour_info.append((
                    c,
                    cv2.isContourConvex(c),
                    cv2.contourArea(c),
                    ))


        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]
        
        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

        #-- Blend masked img into MASK_COLOR background --------------------------------------
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
        frame1        = frame.astype('float32') / 255.0                 #  for easy blending

        masked = (mask_stack * frame1) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')                # Convert back to 8-bit
        masks.append(masked)
        cv2.imshow('img', masked)                                   # Display
        cv2.waitKey(1)
    if(len(masks)==0):
        return 0
    masked = masks[0]
##    print("I am ere")
    for mask in masks[1:]:
        masked = cv2.bitwise_or(masked,mask)
    return masked

