# import the necessary packages
import math
from os import truncate
import numpy as np
import pickle
import cv2
from skimage.feature import hog,local_binary_pattern
# load the actual face recognition model along with the label encoder
print("[INFO] loading model")
recognizer = pickle.loads(open("classifier.pickle", "rb").read())
#pca = pickle.loads(open("pca.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())
print("[INFO] Model loaded successfully")

cap = cv2.VideoCapture('c:\\temp\\Junction2.avi')

print("[INFO] Starting with video")
ic = 0
jc = 0
eps=1e-7
numPoints = 24
radius = 8

ret = True
frame=0
data=[]
frames=[]

while (ret):
    ret,img = cap.read()
    if (ret==False):
        break
    img = cv2.resize(img,(800,600))
    roi = img[80:435,270:670]
    col = roi.copy()
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    h,w = roi.shape
    
    ic+=1

    #LV ; this onlywas  shows the first 150 frames in a different window ,for reference
    #if(ic <= 150):
    #    cv2.imshow('temp',roi)
    #    cv2.waitKey(1)
    #    continue

    # every 25*60 = minute , display something for feeling of progress.. 
    if(ic%1500 == 0):
        print(ic)
    
    #take 1 frame per sec ; there are 25 fps. 
    if(ic%25 == 0):
    #if(True):
        red=0
        green=0
        for i in range(44,h,44):
            for j in range(44,w,44):
                box = roi[i-44:i,j-44:j]
                lbp = local_binary_pattern(box, numPoints, radius, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    			# normalize the histogram
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)
 
                lbp_embedding = hist

                # LV  : Replaced multichannel=false with channel_axis=None according documentation. 

                hog_embedding = hog(box, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(1, 1), visualize=False, channel_axis=None)
                embedding = np.append(hog_embedding.ravel(),lbp_embedding)
    			#embedding = pca.transform(embedding.reshape(1, -1))  
                prediction = recognizer.predict(embedding.reshape(1, -1))
    			#cv2.rectangle(nroi,(j,i),(j-121,i-121),(255,0,0),2)

                #LV : changed intensity from 255 to 125 for red and green colors.

                if(prediction == 1):
                    cv2.rectangle(col,(j,i),(j-39,i-39),(0,0,125),1)
                    red = red+1
                else:
                    cv2.rectangle(col,(j,i),(j-39,i-39),(0,125,0),1)
                    green=green+1
                jc+=1

        
        #put percentage of red on top of the image
        ratio = red / (red+green)
        cv2.putText(col,str(round(ratio*100,1)) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,200,200),1)


        #save stats of each frame
        data.append( { "frame":frame, "red": red, "green": green } )
        
        #store frame in memory.. we later save it as frame to a video.
        frames.append(col)

        #cv2.imshow('temp', img)
        #cv2.imshow('temp2',col)
        #cv2.waitKey(1)
        frame = frame +1

f = open("framedata.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

#construct the 'sorted' video
newlist = sorted(data, key=lambda d: d['red']) 

# LV.. yep size is reversed :( first height then width.. Took me 30 min of my life.
size = (400,355)
writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size)

for datarow in newlist:  
    framenumber = datarow['frame']
    writer.write(frames[framenumber])

writer.release()

cv2.destroyAllWindows()


