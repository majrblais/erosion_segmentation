#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "6"
#mport keras
#from keras.utils import generaic_utils
import segmentation_models as sm

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #Pour avoir les GPUs dans le bon orde.
os.environ["CUDA_VISIBLE_DEVICES"] = '4'    #C hoix du GPU.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #Permet d'eviter les erreurs bizarres qui parlent de convolutions



BACKBONE ='densenet121'
CLASSES = ['trees']
#LR = 0.000000000101
LR = 0.000001
preprocess_input=sm.get_preprocessing(BACKBONE)

x_train_dir = "./new_t/images/"
y_train_dir = "./new_t/masks/"

import cv2 as cv
import glob



import tensorflow as tf	


x_valid_dir ="./valid_data/images/"
y_valid_dir = "./valid_data/masks/"


data_generator2 = ImageDataGenerator()
data_generator = ImageDataGenerator()



x_generator = data_generator2.flow_from_directory(directory=x_train_dir,target_size=(1024,1024),batch_size=1,seed=42,class_mode=None,classes=None)
y_generator = data_generator.flow_from_directory(directory=y_train_dir,target_size=(1024,1024),batch_size=1,seed=42,class_mode=None,classes=None)


valx_generator = data_generator2.flow_from_directory(directory=x_valid_dir,target_size=(1024,1024),batch_size=1,seed=42,class_mode=None,classes=None)
valy_generator = data_generator.flow_from_directory(directory=y_valid_dir,target_size=(1024,1024),batch_size=1,seed=42,class_mode=None,classes=None) 


def combine_generator(gen1, gen2):
	while True:
		yield(next(gen1), next(gen2))

generator = combine_generator(x_generator,y_generator)
val_generator = combine_generator(valx_generator,valy_generator)

model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]



model.compile(optim,total_loss,metrics)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

checkpoint_path='./we_'+BACKBONE+'.ckpt'
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,monitor='val_loss',save_best_only=True,verbose=1)
#model.load_weights('./we_densenet121.ckpt')
model.fit(generator,steps_per_epoch=500,epochs=100,validation_data=val_generator,validation_steps=100,callbacks=[cp_callback])

from os import listdir
from os.path import isfile,join

#final=model.predict(testx_generator,verbose=1,steps=21)
#print("FPN_"+BACKBONE)
