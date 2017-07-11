### Import the necessary packages

import numpy as np
import argparse
import imutils
import cv2
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

#########################################################################################################
### TRAINING THE CLASSIFIER

### Class Labels: 
### BIKE - '1'
### HORSE - '2'

### Lists to hold images and class labels wrt input images
image_list =[]
classes = []

### Reading the labeled data {/Path/{class}.{number}.jpg} from the train folder
for paths in glob.glob('/home/akanksha/Desktop/MP/Assignment/Assignment 2/Dataset/Train/*.jpg'):
	im = cv2.imread(paths)
	### Extracting the class labels from the path
	l = paths.split(os.path.sep)[-1].split(".")[0]
	classes.append(l)
	image_list.append(im)

#print(classes)

### CREATING THE SIFT FEATURE DETECTOR

sift = cv2.xfeatures2d.SIFT_create()

features = []

### SIFT Feature extraction

for img in image_list:
	kp, des = sift.detectAndCompute(img, None)
	features.append(des)

# Stack all the descriptors vertically in a numpy array
descriptors = features[0]
for descriptor in features[1:]:
	descriptors = np.vstack((descriptors, descriptor))

#print(descriptors.shape)

### CLUSTERING USING KMEANS
### Specfifying the criteria, number of clusters and the Initalized cluster centers

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1.0)
K = 40
flags = cv2.KMEANS_RANDOM_CENTERS
ret, labels, centers = cv2.kmeans(descriptors, K, None, term_crit, 10, flags)


#print(labels.shape)
#print(centers.shape)

### Calculate HISTOGRAM of features
### Vector Quantization (vq): assign codes from a code book to observations.
### Assigns a code from a code book to each observation. 
### Each observation vector in the M by N obs array is compared with the centroids 
### in the code book and assigned the code of the closest centroid.
im_features = np.zeros((len(image_list), K), "float32")
for i in range(len(image_list)):
    v_words, distance = vq(features[i],centers)
    for w in v_words:
        im_features[i][w] += 1

#print(v_words.shape)
#print(distance.shape)
#print(im_features.shape)
#print(im_features)


### Scaling the Visual Words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

#print(stdSlr)
#print(im_features.shape)

###  Converting the class labels list into a numpy array
trainL = np.array(classes)

### Partition the data into training and testing splits, using 85%
### of the data for training and the remaining 15% for testing
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	im_features, classes, test_size=0.15, random_state=42)


### Training the Logistic Regression Classifier

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(trainFeat, trainLabels)

### Predicting the accuracy 
acc = logreg.score(testFeat, testLabels)
print("[INFO] Accuracy: {:.2f}%".format(acc * 100))
print("Classifier Trained!!")

##############################################################################################
### TESTING THE CLASSIFIER

print("Testing the Classfier!!")

im = cv2.imread(('/home/akanksha/Desktop/MP/Assignment/Assignment 2/Dataset/Test/test_1.jpg'))
    
des_list = []
kpts, des = sift.detectAndCompute(im, None)
des_list.append(des)

descriptors = des_list[0]
for descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

test_features = np.zeros((1, K), "float32")
#print(test_features.shape)
words, distance = vq(des_list[0], centers)
#print(words.shape)
#print(distance.shape)
#print(test_features.shape)

for w in words:
	test_features[0][w] += 1


### Scale the features
test_features = stdSlr.transform(test_features)
#print(test_features.shape)

# Infer and print the prediction
prediction =  logreg.predict(test_features)
print("The corresponding class of the test image is: ")

if prediction == '2':
	text = 'HORSE'
else:
	text = "BIKE"
print(text)

### Printing the results on the output image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
pt = (0, 3 * im.shape[0] // 4)
cv2.putText(im, text, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
cv2.imwrite("Output2.jpg", im)





