#!/usr/bin/env python

import numpy as np
import cv2

data_folder = "./data"

angle_bias = 1.0

def reset_bias():
    global angle_bias
    angle_bias = 1.0

def brightness(X):
    tmp = cv2.cvtColor(X, cv2.COLOR_RGB2YUV)
    tmp[:, :, 0] = cv2.equalizeHist(tmp[:, :, 0])
    return cv2.cvtColor(tmp, cv2.COLOR_YUV2RGB)

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
        
    # Remove the top part
    X = X[60:140, :, :]

    if resize:
        X = cv2.resize(X, (64, 64), interpolation=cv2.INTER_AREA)
    
    return X

def prepare(X, y):
    X = preProcess(X, y)

        # Translate image to horizontal and vertical direction
    X, y = translate(X, y, 100, 40)

    # With a random probability miror the image from left to right, 
    mirror = np.random.randint(2)
    if mirror:
        X = np.fliplr(X)
        y = -y
    
    return X, y

def training_generator(data, steering, batch_size=128):
    global angle_bias
    
    new_shape = (64, 64)
    batch_X = np.zeros((batch_size, new_shape[0], new_shape[1], 3))
    batch_y = np.zeros(batch_size)
    cnt = 0
    while 1:
        while (cnt < batch_size):
            # Randomly select an image. 
            index = np.random.randint(len(data))
            camera = np.random.randint(3)

            file = data[index][camera]
            shift = [0, .2, -.2] #shift angle for center, left, and right camera
            image = cv2.imread(file)
            y = steering[index] + shift[camera]
            
            X, y = prepare(image, y)
            
            # Higher bias value will pick smaller angles. 
            threshold = np.random.uniform()
            if abs(y) + angle_bias < threshold:
                continue

            X = cv2.resize(X, new_shape, interpolation=cv2.INTER_AREA)

            batch_X[cnt] = X
            batch_y[cnt] = y

            cnt += 1
        angle_bias = angle_bias / 2
        yield (batch_X, batch_y)