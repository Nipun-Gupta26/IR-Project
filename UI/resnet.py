import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class Encoder():
  def __init__(self, model):
    self.model = model
  
  def encode(self, img_path):

   #get the image path
    img = image.load_img(img_path, target_size=(224, 224, 3)) #load the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) # convert to (1, 224, 224, 3) shaped nd-array

    ftr_vector = model.predict(x) #get the output as (batch_size, 7, 7, 2048)
    return ftr_vector
  
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224,3), classes = 1000, classifier_activation = False)    
enc = Encoder(model)

ftr_vec = enc.encode('1.jpg')
print(ftr_vec.shape)