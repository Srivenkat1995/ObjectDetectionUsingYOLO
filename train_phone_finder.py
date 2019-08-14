import numpy as np
from darkflow.net.build import TFNet
import cv2
import xml.etree.cElementTree as ET
import sys
import os
import string
import glob
import shutil

###################################################################################################################################
#       References : 
#                       1. https://github.com/thtrieu/darkflow
#                       2. https://pjreddie.com/darknet/
#
#
###################################################################################################################################



###################################################################################################################################
#
#  The method getSourceDirtectory() is used to get the parent directory of the given path.
# 
#  Example : let us assume the path be  a/b/c , the output is a/b
#   
#   
#
###################################################################################################################################

def getSourceDirectory(path):

    paths = path.split("/")
    new_path = ""
    for pa in range(len(paths) - 2):
        new_path = new_path + paths[pa] + '/'
    return new_path

##################################################################################################################################

# The method createAnnotationFile() creates an xml file with the details about the label and bounding box coordinates which is 
# used by YOLO Model. 

###################################################################################################################################


def createAnnotationFile(filename, centerX, centerY, path):

    path_updated = path + filename
    image = cv2.imread(path_updated)
    rows,columns,channels = image.shape
    width = str(rows)
    height = str(columns)
    depth = str(channels)
    xmin = round((float(centerX) * int(height)) - 20)  
    ymin = round((float(centerY) * int(width)) - 20)
    xmax = round((float(centerX) * int(height)) + 20)
    ymax = round((float(centerY) * int(width)) + 20) 
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "find_phone"
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = path
    source = ET.SubElement(root, "source")
    ET.SubElement(source,"database").text = "Unknown"
    size = ET.SubElement(root,"size")
    ET.SubElement(size,"width").text = height
    ET.SubElement(size,"height").text = width
    ET.SubElement(size,"depth").text = depth
    ET.SubElement(root,"segmented").text = "0"
    object = ET.SubElement(root,"object")
    ET.SubElement(object,"name").text = "phone"
    ET.SubElement(object,"pose").text = "unspecified"
    ET.SubElement(object,"truncated").text = "0"
    ET.SubElement(object,"difficult").text = "0"
    bndbox = ET.SubElement(object,"bndbox")
    ET.SubElement(bndbox,"xmin").text = str(xmin)
    ET.SubElement(bndbox,"ymin").text = str(ymin)
    ET.SubElement(bndbox,"xmax").text = str(xmax)
    ET.SubElement(bndbox,"ymax").text = str(ymax)
    tree = ET.ElementTree(root)
    sourceDirectory = getSourceDirectory(path)
    files = filename.split(".")
    tree.write(sourceDirectory + "annotations/" + files[0] + ".xml")

##############################################################################################################################
# The main function which trains the YOLO Model
##############################################################################################################################


if __name__== "__main__":

    fn = sys.argv[1]

############################################################################################################################
#       Iterates through the files in the given path and parses the label.txt file and gets the bounding box coordinates 
#       with the given the center coordinates.
############################################################################################################################

    directory = os.fsencode(fn)
    print(directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        dir = 'find_phone/' + filename 
        if filename == "label.txt": 
            with open(dir) as f:
                content = f.readlines()
            content = [x.strip() for x in content] 
            
    for iterate in content:
        value = iterate.split(" ")
        createAnnotationFile(value[0],value[1],value[2],fn)

################################################################################################################################
#  Copies the image files from the given path directory to another directory for input to the YOLO Model
################################################################################################################################

    src_dir = fn 
    dst_dir = getSourceDirectory(fn) + "phone_images_directory/"
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(jpgfile, dst_dir)
    options = {"model": "cfg/yolo_custom_phone.cfg", 
           "load": "yolo.weights",
           "batch": 8,
           "epoch": 125,
           "train": True,
           "annotation": "annotations/",
           "dataset": "phone_images_directory/"}

##################################################################################################################################
# Training the model by passing the annotations directory, images_directory, cfg file and weights file 
##################################################################################################################################

    tfnet = TFNet(options)

    tfnet.train()

    tfnet.savepb()

    print("The training Process is complete")

#################################################################################################################################3
#  Training process is complete. Now running the find_phone.py will give the results
# ################################################################################################################################    

