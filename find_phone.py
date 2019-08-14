import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
import os
import sys
import math


#####################################################################################################################################

#  The method predictResult() is used to convert the output from the YOLO Model to the ones which we desire i.e the normalized 
#  coordinates of the phone. Here, the output of the YOLO Model predicts the diagonal coordinates of the probable bounding box 
#  around which the phone is detected. In an ideal scenario, Finding the mid point of the diagonal line formed by the two coor
#  -dinates say P(xmin,ymin) and Q(xmax, ymax) gives me the center of the bounding box which is in turn the center of the phone.
#  This is how we find the coordinates of the phone in the image. The coordinates are in the form of pixel positions. We now do
#  a simple division with the corresponding dimensions of the image to get the normalized coordinates.     

#####################################################################################################################################

def predictResult(results,columns,rows):

    
    if results is not None:
        
        for result in results:
        
            top_x = result['topleft']['x']
            
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            
            label = result['label'] + " " + str(round(confidence, 3))

            distance = math.sqrt(math.pow((btm_x - top_x),2) + math.pow((btm_y - top_y),2))
            
            center = round(distance/2)
            
            centerX, centerY = (top_x + center)/columns, (top_y + center)/rows
            
            print("{0:.4f}".format(centerX),"{0:.4f}".format(centerY)) 
    
#######################################################################################################################################

# The Magic box which outputs the normalized coordinates of the phone given the imput image.  

#######################################################################################################################################


if __name__== "__main__":

########################################################################################################################################

#  To get input path to the image 

########################################################################################################################################

    fn = sys.argv[1]
    
    image_input = cv2.imread(fn)

    original_img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    rows, columns, channels = original_img.shape

    if image_input is None:
        print (" Enter the correct Image Path")

##################################################################################################################################

# Since we have already run the python script, train_phone_finder.py, the training process of the model is stored as artifacts.
# Now, its just loading the trained model and testing it on top of our image. 

##################################################################################################################################

    options = {"model": "cfg/yolo_custom_phone.cfg",
             "load": -1,
             "gpu": 1.0 }
    
    tfnet = TFNet(options)
    
    tfnet.load_from_ckpt()
    
    results = tfnet.return_predict(original_img)

####################################################################################################################################

# Now, Converting the predicted results to normalized coordinates of the phone location in the image

####################################################################################################################################

    predictResult(results,columns,rows)
        
    


    