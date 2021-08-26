from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle



# re-size all the images to this
X = pickle.load(open('X.p','rb'))
y = pickle.load(open('y.p','rb'))

y = np.array(y)

X = X/255.0

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(monitor='val_loss', patience=2)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

# add preprocessing layer to the front of PRE-TRAINED MODEL
baseModel = VGG16(input_shape=X.shape[1:], weights='imagenet', include_top=False)

# don't train existing weights
for layer in baseModel.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('C:/Users/Nova DC/Desktop/Drowsy_statedetection/drowsysystem/Bhutanese Drowsiness Dataset1/training/*')

# our layers - you can add more if you want
x = Flatten()(baseModel.output)
#x = Dense(128, activation='relu')(x)
#x = Dropout(0.5)(x)
headModel = Dense(1, activation='sigmoid')(x)

# create a model object
model = Model(inputs=baseModel.input, outputs=headModel)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# fit the model
r = model.fit(X, y, batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks)

# loss
plt.style.use("ggplot")
plt.figure()
plt.title('Cross Entropy Loss')
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig('LossVal_loss.png')

# accuracies
plt.style.use("ggplot")
plt.figure()
plt.title('Accuracy')
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.savefig('AccVal_acc.png')

model.save('vgg3-DrowsyNet.h5')
