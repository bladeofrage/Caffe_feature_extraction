import sys
import glob
import cv2
import numpy as np
from sklearn import svm
from sklearn import metrics

import pathlib  
from pathlib import Path
from datetime import datetime

start_time = datetime.now()

#load all of our data and labels
f4 = np.load('dataset/train_data8m.npy')
train_label = np.load('dataset/train_label.npy')
v4 = np.load('dataset/val_data8m.npy')
val_label = np.load('dataset/val_label.npy')

end_time = datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()
print("Load time in seconds: ", execution_time)
execution_time = time_diff.total_seconds() * 1000
print("Load time in miliseconds: ", execution_time)


#Change and/or set the SVM parameters to finetune it
clf = svm.SVC(kernel='linear')

start_time = datetime.now()

#Train the model using the training sets
clf.fit(f4, train_label.ravel()) #dont split

end_time = datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()
print("training time in seconds: ", execution_time)
execution_time = time_diff.total_seconds() * 1000
print("training time in miliseconds: ", execution_time)


start_time = datetime.now()
#Predict the response for test dataset
val_pred = clf.predict(v4)

end_time = datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()
print("prediction time in seconds: ", execution_time)
execution_time = time_diff.total_seconds() * 1000
print("prediction time in miliseconds: ", execution_time)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(val_label, val_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(val_label, val_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(val_label, val_pred))

