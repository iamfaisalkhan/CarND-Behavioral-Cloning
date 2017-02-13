from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, LeakyReLU
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
    return model

def model_nivida2():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
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

    return model

def model_nivida2b():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))
    model.add(Convolution2D(3, 1, 1,  activation="relu", border_mode="same"))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
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

    return model


def model_nvidia3():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal',  border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def model_nvidia_elu():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal',  border_mode="valid"))
    model.add(ELU())
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('elu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('elu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def model_nvidia_relu():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(36, 5, 5, activation='relu',  subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init='he_normal',  border_mode="valid"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def model_comma_elu():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def model_comma_relu():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

def model_comma_lrelu():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.0 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(LeakyReLU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    #model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(512))
    #model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

