import os
import numpy as np
import cv2

from preprocess import prepareTrain, prepareTest

from config import conf

def training_generator2(data, angle_bias=1.0, batch_size=128):
    batch_X = np.zeros((batch_size, conf.row, conf.col, 3))
    batch_y = np.zeros(batch_size)
    data_folder = conf.data_folder
    cnt = 0
    while 1:
        while (cnt < batch_size):
            # Randomly select an image. 
            index = np.random.randint(len(data))
            camera = ['center', 'left', 'right']
            ci = np.random.randint(3)

            file = os.path.join(data_folder, data[camera[ci]].iloc[index].strip())
            shift = [0, 0.25, -0.25] #shift angle for center, left, and right camera
            
            image = cv2.imread(file)

            y = data['steering'].iloc[index] + shift[ci]
            
            X, y = prepareTrain(image, y)

            # Higher bias value will pick smaller angles. 
            threshold = np.random.uniform()
            if abs(y) + angle_bias < threshold:
                continue

            batch_X[cnt] = X
            batch_y[cnt] = y

            cnt += 1


        yield (batch_X, batch_y)


def training_generator(data, angle_bias=1.0, batch_size=128):
    data_folder = conf.data_folder
    cnt = 0
    while 1:
        while cnt < batch_size:
            # Randomly select an image. 
            index = np.random.randint(len(data))
            camera = ['center', 'left', 'right']
            ci = np.random.randint(3)

            file = os.path.join(data_folder, data[camera[ci]].iloc[index].strip())
            shift = [0, 0.25, -0.25] #shift angle for center, left, and right camera
            
            image = cv2.imread(file)

            y = data['steering'].iloc[index] + shift[ci]
            
            X, y = prepareTrain(image, y)

            # Higher bias value will pick smaller angles. 
            threshold = np.random.uniform()
            if abs(y) + angle_bias < threshold:
                continue
            
            cnt += 1

            # yield X.astype('float32'), y
            yield X, y

def validation_generator(data, batch_size=128):
    batch_X = np.zeros((batch_size, conf.row, conf.col, 3))
    batch_y = np.zeros(batch_size)
    data_folder = conf.data_folder
    cnt = 0
    while 1:
        while (cnt < batch_size):
            # Randomly select an image. 
            index = np.random.randint(len(data))

            file = os.path.join(data_folder, data['center'].iloc[index].strip())

            image = cv2.imread(file)
            y = data['steering'].iloc[index]

            X, y = prepareTest(image, y)

            batch_X[cnt] = X
            batch_y[cnt] = y

            cnt += 1
            
        yield (batch_X, batch_y)