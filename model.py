import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def toHsv(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def v_channel(img):
    v = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    return v.reshape(v.shape + (1,))

def CNNPreProcess():
    model = Sequential()
    model.add (Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,1)))
    model.add (Cropping2D(cropping=((70,25),(0,0))))
    return model

def NvidiaCNN(model):
    model.add (Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add (Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add (Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add (Convolution2D(64,3,3, activation="relu"))
    model.add (Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

samples = []
with open('mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

angle_adjust=[0.0,0.25,-0.25]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for camera in range(3):
                    name = 'mydata/IMG/'+batch_sample[camera].split('/')[-1]
                    image = v_channel(cv2.imread(name))
                    angle = float(batch_sample[3]) + angle_adjust[camera]
                    images.append(image)
                    angles.append(angle)
                    flipped = cv2.flip(image,1)
                    flipped = flipped.reshape(flipped.shape + (1,))
                    images.append(flipped)
                    angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = CNNPreProcess()
model = NvidiaCNN(model)

model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)*6,
                                     nb_epoch=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
