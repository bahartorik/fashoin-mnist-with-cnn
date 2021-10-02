# build and train cnn model for fashion mnist dataset
# this is a simple code for bigener

# import library
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras import layers
from keras import models
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sn
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd

# load data
(image_train, label_train), (image_test, label_test) = fashion_mnist.load_data()

# Flatten the data 
image_train = image_train.reshape(image_train.shape[0],28,28,1) 
image_test = image_test.reshape(image_test.shape[0],28,28,1) 

# build model
model = models.Sequential()

model.add(layers.Conv2D(256, (2, 2), activation='elu',padding='same', input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(layers.Conv2D(256, (2, 2),padding='same', activation='elu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(layers.Conv2D(128, (2, 2),padding='same', activation='elu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Conv2D(128, (2, 2),padding='same', activation='elu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='elu'))
model.add(Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

# see model
model.summary()

# my callback function
thr_acc = 0.92

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > thr_acc): 
            print("\nReached %2.2f%% accuracy, so stopping training!!"%(thr_acc*100))  
            self.model.stop_training = True
            
            
callbacks = myCallback()

# train model
model.compile(loss='sparse_categorical_crossentropy'
,optimizer='adam' ,
metrics=['accuracy'])

history=model.fit(image_train, label_train, batch_size=128, epochs=300, validation_split=0.2, callbacks=[callbacks], verbose=2)

# show loss curve
loss_curve = pd.DataFrame(history.history)
loss_curve.loc[0:, ['loss', 'val_loss']].plot()

# show acc curve
acc_curve = pd.DataFrame(history.history)
acc_curve.loc[0:, ['accuracy', 'val_accuracy']].plot()

# in this part evalutale model
test_loss,test_accuracy = model.evaluate(image_test, label_test)
Y_pred = model.predict(image_test)
Y_pred = np.argmax(Y_pred, axis=1)
