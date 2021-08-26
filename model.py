#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import cv2
import os
from PIL import Image

train_dir = r'C:/Users/Nova DC/Desktop/Drowsy_statedetection/drowsysystem/Bhutanese Drowsiness Dataset1/training'
valid_dir = r'C:/Users/Nova DC/Desktop/Drowsy_statedetection/drowsysystem/Bhutanese Drowsiness Dataset1/validation'
test_dir = r'C:/Users/Nova DC/Desktop/Drowsy_statedetection/drowsysystem/Bhutanese Drowsiness Dataset1/testing'


# In[2]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    zoom_range = 0.2,
    shear_range = 0.2)

val_datagen = ImageDataGenerator(
    rescale = 1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    shuffle=False)


# In[3]:


train_generator.class_indices


# In[4]:


from glob import glob

folders=glob('C:/Users/Nova DC/Desktop/Drowsy_statedetection/drowsysystem/Bhutanese Drowsiness Dataset1/training/*')
print(folders)


# In[5]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(monitor='val_loss', patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[6]:


callbacks = [earlystop, learning_rate_reduction]


# In[7]:


from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu',  padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(layers.Dense(len(folders), activation='sigmoid'))

model.summary()


# In[8]:


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


# In[9]:


history=model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=500,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks)


# In[10]:


plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc') 


# In[11]:


_, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=0)
print('> %.3f' % (acc * 100.0))


# In[12]:


model.evaluate(test_generator)


# In[13]:


# Creating the Confusion Matrix on our test data
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


# In[14]:


model.save('DrowsyNetzero.h5')


# In[ ]:




