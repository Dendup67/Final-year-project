#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import cv2
import os
from PIL import Image


# In[2]:


test_dir = r'C:\\Users\\Nova DC\\Desktop\\Drowsy_statedetection\\drowsysystem\\Bhutanese Drowsiness Dataset1\\testing'
model = tf.keras.models.load_model(r'C:\Users\Nova DC\DrowsyNetzero.h5')
model.summary()


# In[3]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224))

class_names = ['Awake', 'Drowsy']


# In[4]:


_, acc = model.evaluate(test_generator)
print('> %.3f' % (acc * 100.0))


# In[5]:


model.evaluate(test_generator)


# In[6]:


from sklearn.metrics import  classification_report, confusion_matrix

class_names = ['Awake', 'Drowsy']

Y_pred = model.predict(test_generator, 400 // 32+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('--------------')
print('Classification Report')
target_names = ['Awake', 'Drowsy']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# In[7]:


imgs, labels = next(test_generator)
fig=plt.figure(figsize=(8,8))
columns = 4
rows  = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img_t = np.expand_dims(imgs[i], axis = 0)
    prediction = model.predict(img_t)
    idx = prediction[0].tolist().index(max(prediction[0]))
    plt.text(20, 50, class_names[idx], color='red', fontsize=12,bbox=dict(facecolor= 'white',alpha=0.8))
    plt.imshow(imgs[i])


# In[ ]:




