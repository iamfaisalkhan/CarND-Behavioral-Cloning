#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import cv2

from model import *

path = "./data"

def brightness(X, range=0.25):
    tmp = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
    tmp[:, :, 2] = tmp[:, :, 2] + (range + np.random.uniform())
    return cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)

def translate(X, y, x_range, y_range):
    trX = x_range * np.random.uniform() - x_range/2
    trY = y_range * np.random.uniform() - y_range/2
    
    transMat = np.float32([[1, 0, trX], [0, 1, trY]])
    X = cv2.warpAffine(X, transMat, (X.shape[1], X.shape[0]))    
    
    # Translate the steering angle by 0.004 per pixel
    y = y + ( (trX/x_range) * 2 ) *.2
    
    return X, y

def preProcessImage(X, resize=0):
    # Equalize brightness through histogram equlization
    X = brightness(X)
        
    X = X[60:135, :, :]

    return X

def prepare(X, y):
    X = preProcessImage(X)

    # Translate image to horizontal and vertical direction
    X, y = translate(X, y, 100, 40)

    # With a random probability miror the image from left to right, 
    mirror = np.random.randint(2)
    if mirror == 1:
        X = np.fliplr(X)
        y = y * - 1.0
    
    return X, y




if __name__ == "__main__":
    row = 160
    col = 320
    ch = 3

    data = pd.read_csv("%s/driving_log.csv"%path)

    model = getModel_nvidia_original()
    cnt = 0
    for i in range(EPOCHS):
        bias = 1. / (cnt + 1.)
    
        print (bias)
        history = model.fit_generator(
                        training_generator(data, bias, 128), 
                        samples_per_epoch=20000, 
                        validation_data=validation_generator(testing_files, testing_angles, 128),
                        nb_val_samples=1000,
                        nb_epoch=1, 
                        verbose=2)
        cnt +=1
    
    model.save("steering_model_nvidia.h5")
    print ("Model saved")

    