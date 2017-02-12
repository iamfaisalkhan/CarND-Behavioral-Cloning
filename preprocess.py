import cv2
import numpy as np

from config import conf

def brightness(X, range=0.25):
    tmp = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
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

def roi(X):
    top = conf.roi[0][0]
    bottom = conf.roi[0][1]
    left = conf.roi[1][0]
    right = conf.roi[1][1]
   
    return X[top:bottom, left:right, :]

def prepareTrain(X, y):
    X = brightness(X)
    
    X = roi(X)

    # # Translate image to horizontal and vertical direction
    X, y = translate(X, y, 100, 40)

    # # With a random probability miror the image from left to right, 
    mirror = np.random.randint(2)
    if mirror == 1:
        X = cv2.flip(X, 1)
        y = y * - 1.0
    
    X = cv2.resize(X, (conf.row, conf.col), cv2.INTER_AREA)

    return X, y

def prepareTest(X, y):
    #X = roi(X)

    #X = cv2.resize(X, (conf.row, conf.col), cv2.INTER_AREA)

    #X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

    return X, y


