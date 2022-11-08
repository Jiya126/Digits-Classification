from copyreg import pickle
import importlib
from importlib.resources import path
from multiprocessing import pool
from operator import mod
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout
from keras.optimizers import Adam
import pickle
from pickle import dump

images =[]
classNo = []
path = 'myData'
testRatio = 0.2
validRatio = 0.2
imgDim = (32,32,3)
batchSizeVal = 50
epochVal = 100
# stepsPerEpochVal = 2000

myList = os.listdir(path)
noOfClasses = len(myList)

print('Importing classes.....')
for x in range(0, noOfClasses):
    myPicList = os.listdir(path +'/' +str(x))
    for y in myPicList:
        curImg = cv2.imread(path +'/' +str(x) +'/' +y)
        curImg = cv2.resize(curImg, (imgDim[0], imgDim[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")

images = np.array(images)
classNo = np.array(classNo)


# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validRatio)

noOfSamples = []
for x in range(0, noOfClasses):
    noOfSamples.append(len(np.where(y_train==x)[0]))

# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(y_test))

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), noOfSamples)
plt.title('Training Images Data')
plt.xlabel('no of classes')
plt.ylabel('no of samples')
plt.show()

# PRE-PROCESSING
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_valid = np.array(list(map(preProcess, x_valid)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)

# IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.2,
shear_range=0.1,
rotation_range=10)

dataGen.fit(x_train)  # generates statistics  

# ONE HOT ENCODING
# generates list of 1,0,0..... with 1 on class that is being shown
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_valid = to_categorical(y_valid, noOfClasses)

# MODEL
def myModel():
    noOfFilters = 60
    filter1 = (5,5)
    filter2 = (3,3)
    poolSize = (2,2)
    noOfNodes = 500

    # conv2D, conv2D, MaxPooling, conv2D, conv2D, MaxPooling, Dropout, Flatten, Dense, Dropout, Dense

    model = Sequential()
    model.add(Conv2D(noOfFilters, filter1, input_shape=(imgDim[0], imgDim[1],1), activation='relu'))
    model.add(Conv2D(noOfFilters, filter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=poolSize))
    model.add(Conv2D(noOfFilters//2, filter2, activation='relu'))
    model.add(Conv2D(noOfFilters//2, filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=poolSize))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# TRAINING PROCESS
history = model.fit(dataGen.flow(x_train, y_train, batch_size=batchSizeVal), epochs= epochVal, validation_data=(x_valid, y_valid), shuffle=1)     #fit to train model, flow returns bathes of augmented images


# PLOT TRAINED DATA
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


# EVALUATE USING TEST IMAGES
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])


# SAVE MODEL
pickle_out = open('model_trained.p', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()