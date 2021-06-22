#sources used to create this code:
#for caffe feature extraction
#https://prateekvjoshi.com/2016/04/26/how-to-extract-feature-vectors-from-deep-neural-networks-in-python-caffe/
#for channelwise image mean
#https://github.com/S-Mahdi-Hosseini/RGB_images_mean_and_std/blob/master/RGB_normalization.py
import os
import sys
import glob
import cv2
import imageio
import numpy as np
#from sklearn import svm
#from sklearn import datasets
#from sklearn import metrics
from PIL import Image
import pathlib  
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '/home/gijs/caffe-master/python') #path your caffe master folder
import caffe

caffe.set_mode_cpu() #We're only using the CPU

#using code from https://github.com/S-Mahdi-Hosseini/RGB_images_mean_and_std/blob/master/RGB_normalization.py to
#compute the channelwise imagemean. This is used to give the CNN additional information of the image set we are using
def imgs_mean(*args):
  #use glob imported from glob module to create a list of image path and give this list as arguments of this function
  img_list = []
  for lists in args:
    img_list = img_list + list(lists)
  mean = [0 , 0 , 0]  # [R , G , B]
  for i , img in enumerate(img_list):   # shapes of images are   H * W * D
    image = cv2.imread(img) #Using cv2 imread instead of pyplot imread to ensure we have RGB channels even when reading
                            #a greyscale image.
    mean[0] = (mean[0]*i + np.mean(image[:,:,0]))/(i+1)
    mean[1] = (mean[1]*i + np.mean(image[:,:,1]))/(i+1)
    mean[2] = (mean[2]*i + np.mean(image[:,:,2]))/(i+1)
  return mean


#this generates the labels, needs to only happens once for the training set, and one more time for the test/evaluation set
#when using more than 1 class you're going to want to use a different method of generating the labels.
#Once that is done you can comment out these 2 lines.
label_array = np.vstack((np.ones([1300, 1],int),np.zeros([1300,1],int))) #change the number 1300 to the number of images you for each class
np.save('dataset/train_label.npy', label_array)

#load images of coral
image_list = []
dim = (224, 224) #set the dimension so that each image is uniform in size/shape. Size can vary on your CNN
#you can glob these together but to make absolutely sure we know which set we label with 1 and wich with 0, we do them separately
for filename in glob.glob("/home/gijs/Documents/bachelor/code/dataset/real_test/n01917289/*.JPEG"): #glob for a list of the files
  image = cv2.imread(filename)
  #resize
  image = cv2.resize(image, dim)
  #add images to list
  image_list.append(image)

#repeat for anemone
for filename in glob.glob("/home/gijs/Documents/bachelor/code/dataset/real_test/n01914609/*.JPEG"): #glob for a list of the files
  image = cv2.imread(filename)
  #resize
  image = cv2.resize(image, dim)
  #add images to list
  image_list.append(image)
#Because we read the images for corral first, corral has the label 1 and brain anemone has label 0



#______________________________IMAGENET__________________________________________________________________________________
model_file = '/home/gijs/Documents/bachelor/code/devilindetailmodel/VGG_CNN_S.caffemodel'  #path to model
deploy_prototxt = '/home/gijs/Documents/bachelor/code/devilindetailmodel/VGG_CNN_S_deploy.prototxt' #path to prototext
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST) #initialise the net


#if Path.exists('mean.npy'):
#  print("Loading mean")
#  imagemean = np.load('mean.npy')
#else:
#  print("Generating mean...")
#  imagemean = np.array(imgs_mean(glob.glob("/home/gijs/Documents/bachelor/code/dataset/training/**/*.JPEG")))
#  print("Saving mean as 'mean.npy' in current directory")
#  np.save('mean.npy', imagemean)
imagemean = np.array(imgs_mean(glob.glob("/home/gijs/Documents/bachelor/code/dataset/training/**/*.JPEG"))) #generate the mean

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #set up the net
transformer.set_mean('data', imagemean)     #set the image mean
transformer.set_transpose('data', (2,0,1))  #This step with these values is practically mandatory. Caffe wants image data to be parsed as Channel_Height_Width
                                            #but most of pythons image reading tools read images as Heigth_Width_Channel, this function converts H_W_C to C_H_W
transformer.set_raw_scale('data', 255.0)    #This sets the range of RGB values, python usually has a range of [0, 1] but many models use the [0, 256] range

#if needed you can reshape the network blob to the shape needed for the current CNN architecture.
#net.blobs['data'].reshape(1,3,224,224)


#extract features from 2 dimensional layers, these features need to be flattened.
#Caffe generally uses the names conv1 through conv5 and fc6 through fc8 for their layers.
#comment this out if you are extracting 1-dimensional features
layer2d = 'conv5' #name of layer you want to extract from.
if layer2d not in net.blobs:
  raise TypeError("Invalid layer name: " + layer2d)
features_array = np.zeros((2,net.blobs[layer2d].data[0].flatten().size))#need to initialise 2D array else new entries will be insterted on the same row and not a new one
i = 0
for img in image_list:
  net.blobs['data'].data[...] = transformer.preprocess('data', img) #prepare image for the network
  output = net.forward() #this pushes the image through the network
  #first 2 entries need to be done manually therefor an if else statement for i < 2 and i >= 2
  if i < 2:
    features_array[i] = net.blobs[layer2d].data[0].flatten() #this works for the first one
  else:
    features_array = np.insert(features_array, i, net.blobs[layer2d].data[0].flatten(), axis = 0)
  i = i+1
np.save('dataset/val_data5m.npy', features_array) #save the array to disc

#extract features from 1-dimensional layers, these don't need to be flattened
#comment this out if you are extracting 2-dimensional features
layer1d = 'fc6' #name of layer you want to extract from.
if layer1d not in net.blobs:
  raise TypeError("Invalid layer name: " + layer1d)
features_array = np.zeros((2,net.blobs[layer1d].data[0].size))#need to initialise 2D array else new entries will be insterted on the same row and not a new one
i = 0
for img in image_list:
  net.blobs['data'].data[...] = transformer.preprocess('data', img)
  output = net.forward()
  #first 2 entries need to be done manually therefor an if else statement for i < 2 and i >= 2
  if i < 2:
    features_array[i] = net.blobs[layer1d].data[0] #this works for the first one
  else:
    features_array = np.insert(features_array, i, net.blobs[layer1d].data[0], axis = 0)
  i = i+1
np.save('dataset/val_data6m.npy', features_array) #save the array to disc

exit()

