import os
import cv2
import numpy as np

from skimage.exposure import rescale_intensity

from config import conf

def gray_diff_images(data):
    rows = data.shape[0]
    X = np.zeros((rows - 6, conf.h, conf.w, 4), dtype=np.uint8)
    for i in range(4+2,  rows):
        for j in range(4):
            file = os.path.join(conf.data_folder, data['center'].iloc[i - j - 1])
            img1 = cv2.imread(file)
            file = os.path.join(conf.data_folder, data['center'].iloc[i - j - 1])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2 = cv2.imread(data['center'].iloc[i - j])
            img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img = img2[:, :, 2] - img1[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))

            img = np.array(img, dtype=np.uint8)
            X[i - 6, :, :, j] = img

    return X, np.array(data["steering"].iloc[6:])

def brightness(X, val=0.25):
    tmp = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
    bgt = val + np.random.uniform()
    tmp[:, :, 2] = tmp[:, :, 2] * bgt

    return cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)

def translate(X, y, x_range, y_range):
    trX = x_range * np.random.uniform() - x_range/2
    trY = y_range * np.random.uniform() - y_range/2
    
    transMat = np.float32([[1, 0, trX], [0, 1, trY]])
    X = cv2.warpAffine(X, transMat, (X.shape[1], X.shape[0]))    
    
    # Translate the steering angle by 0.004 per pixel
    y = y + ( (trX/x_range) * 2 ) *.2
    
    return X, y

def rotate(X, y, rot_angle):
    ang_rot = np.random.uniform(rot_angle)-rot_angle/2
    rows,cols,_ = X.shape    
    rot_M = cv2.getRotationMatrix2D((cols/2,rows/2), rot_angle, 1)
    X = cv2.warpAffine(X, rot_M,( cols,rows) )

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
    
    X = cv2.resize(X, (conf.col, conf.row), cv2.INTER_AREA)

    return X, y

def prepareTest(X):

    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

    X = roi(X)

    X = cv2.resize(X, (conf.col, conf.row), cv2.INTER_AREA)

    return X

