from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D

from config import conf

def model_nivida1():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu", border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Created and compiled Nvidia1 model')

    return model

def model_nivida2():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Created and compiled Nvidia\'s original model')

    return model
