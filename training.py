# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dropout

from keras.callbacks import TensorBoard
import time

# NAME = "image-classification-CNN-64x2-{}".format(int(time.time()))

# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

trainDirectory = './hw4_train'
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(trainDirectory, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            training_data.append([img_array, class_num])

# def create_training_data():
#     for category in categories:
#         path = os.path.join(trainDirectory, category)
#         class_num = categories.index(category)
#         for i in range(0, 3000):
#             img = str(class_num) + '_' + str(i) + '.png'
#             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#             training_data.append([img_array, class_num])

create_training_data()

import random
random.shuffle(training_data)

train_x = []
train_y = []

for features, label in training_data:
    train_x.append(features)
    train_y.append(label)

train_x = np.array(train_x).reshape(-1, 28, 28, 1)
train_y = np.array(train_y)

train_x = train_x/255.0

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# NAME = "{}".format(int(time.time()))    
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = train_x.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(200))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=64, epochs=20, validation_split=0.1, callbacks=[callback])
# model.fit(train_x, train_y, batch_size=64, epochs=6, validation_split=0.0)

model.save('image-classification-CNN-64x2.model')