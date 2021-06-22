# Caffe_feature_extraction
a program to extract features from CNNs in Caffe

This is part of a thesis made at the LIACS faculty of the Leiden University.
Anyone is free to use this code.

# Instructions
## Installing Caffe
First you need to acquire the Caffe library. Our code was made for Ubuntu so 
that installation is what we are following:
https://caffe.berkeleyvision.org/install_apt.html
For ubuntu 17.04 and higher there is a apt-get command you can follow for the
installation of Caffe and another for installing the dependencies. I tried this 
once on a virtual machine and couldn't run my code, so your results may vary.

I recommend just sticking with the manual download of Caffe from their github 
and following the instructions for Ubuntu 16.04 and lower. Be mindful that not
all the steps are in the white boxes on the website. A few of the steps are in
the text surrounding the boxes, these can be easily overlooked.

## Running the program
Once you got Caffe succesfully installed, the programs can be found on:
github link here
The programs need to be run from a terminal or command line, navigate to the
directory where the programn is saved execute these commands:  
python3 extraction.py  
python3 svm.py  
  
python3 extraction.py for running the feature extraction  
python3 svm.py to train an SVM on the features.  

The code contains comments and paths that you need to edit to satisfy your 
paths/folder structure. The code does not support multiclass classifiers. You
need to change how the labels are generated to make sure more there are more
than 2 different labels. You also need to write your own code for the
classifier or alter the existing SVM one so that it supports multiclass.

With the current setup you want to you folder structure to be something like:  
working directory  
|-dataset  
|---training set  
|-----images of class 1  
|-----images of class 2  
|---evaluation set  
|-----images of class 1  
|-----images of class 2  
  
There is also svm_with_PCA_graph.py if you want to look at a visualisation of
the data, but since this alters the data it does not show the metrics
