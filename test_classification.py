from turtle import st
import cv2
import numpy as np
import pickle

thresh = 0.8

c = cv2.VideoCapture(0)

# LOAD TRAINED MODEL
pickle_in = open('model_trained.p', 'rb')
model = pickle.load(pickle_in)

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    s, img = c.read()
    img1 = np.asarray(img)
    img1 = cv2.resize(img1, (32,32))
    img1 = preProcess(img1)
    img1 = img1.reshape(1,32,32,1)

    # PREDICT
    # classIndex = int(model.predict_classes(img))
    predict = model.predict(img1)
    classIndex = int(np.argmax(predict))

    predictions = model.predict(img1)
    probabVal = np.amax(predictions)

    if probabVal > thresh:
        cv2.putText(img, str(classIndex) + " " + str(probabVal), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

    cv2.imshow('classify', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break