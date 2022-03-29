import cv2
from cv2 import ROTATE_180
from cv2 import ROTATE_90_COUNTERCLOCKWISE
from cv2 import ROTATE_90_CLOCKWISE
import numpy as np
from imutils import paths
from skimage.feature import hog,local_binary_pattern
import os
import pickle

imagePaths = list(paths.list_images("dataset"))

Extracted_Names = []
Extracted_Embeddings = []

eps=1e-7
numPoints = 24
radius = 8

for (i, imagePath) in enumerate(imagePaths):
    #if(int(i) >= 10000 and int(i) <= 15000):
    #    continue
    #print("[INFO] processing image {} - {}/{}".format(imagePath, i + 1,len(imagePaths)))
    parts = imagePath.split(os.path.sep)
    filename = parts[-1]
    foldername = parts[-2]
    img = cv2.imread(imagePath)

    img2 = cv2.rotate(img, rotateCode=ROTATE_90_CLOCKWISE)
    img3 = cv2.rotate(img, rotateCode=ROTATE_180)
    img4 = cv2.rotate(img, rotateCode=ROTATE_90_COUNTERCLOCKWISE)

    name2 = "dataset\\big dataset\\{}\\a_2_{}".format(foldername,filename)
    cv2.imwrite(name2,img2)

    name3 = "dataset\\big dataset\\{}\\a_3_{}".format(foldername,filename)
    cv2.imwrite(name3,img3)

    name4 = "dataset\\big dataset\\{}\\a_4_{}".format(foldername,filename)
    cv2.imwrite(name4,img4)


print("done")