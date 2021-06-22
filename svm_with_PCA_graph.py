import sys
import glob
import cv2
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pathlib  
from pathlib import Path


f4 = np.load('dataset/train_data8m.npy')
train_label = np.load('dataset/train_label.npy')
v4 = np.load('dataset/val_data8m.npy')
val_label = np.load('dataset/val_label.npy')


pca = PCA(n_components=2)
features_reduced = pca.fit_transform(f4)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#Create a svm Classifier
model = svm.SVC(kernel='linear') 

#Train the model using the training sets
clf = model.fit(features_reduced, train_label.ravel())

#plot
fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = features_reduced[:, 0], features_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=train_label[:,0], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decison surface using the PCA transformed/projected features')
ax.legend()
plt.show()

