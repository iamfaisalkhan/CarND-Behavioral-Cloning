#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import cv2

from keras.models import load_model

import argparse

from model import *
from generator import training_generator
from generator import validation_generator
from config import conf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Behavior Clonning')

    parser.add_argument(
        '--epochs',
        dest="epochs",
        type=int,
        nargs='?',
        default=10,
        help='Number of training iterations or epochs'
    )
    parser.add_argument(
        '--model',
        dest="model",
        type=str,
        nargs='?',
        default='nvidia1',
        help='Number of training iterations or epochs'
    )
    parser.add_argument(
        '--init',
        dest="model_init",
        type=str,
        nargs='?',
        default=None,
        help='Initialize the model with weights from the file.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='./data',
        help='Path to image folder.'
    )
    parser.add_argument(
        'model_dir',
        type=str,
        nargs='?',
        default='/data',
        help='Path to image folder. This is where the trained model will be saved.'
    )

    args = parser.parse_args()
    
    conf.epochs = args.epochs
    conf.data_folder = args.image_folder
    conf.epochs = args.epochs
    conf.model_dir = args.model_dir
    conf.model = args.model

    data = pd.read_csv("%s/driving_log.csv"%conf.data_folder)
    mask = np.random.rand(data.shape[0]) < 0.9
    train = data[mask] 
    valid = data[~mask]

    if conf.model == "nvidia1":
        model = model_nivida1()
    else:
        print("Model not defined")
        exit()
    
    if args.model_init != None and os.path.exists(args.model_init):
        model = load_model(args.model_init)

    cnt = 0
    for i in range(conf.epochs):
        bias = 1. / (cnt + 1.)
    
        history = model.fit_generator(
                        training_generator(train, bias, 128), 
                        samples_per_epoch=25600, 
                        validation_data=validation_generator(valid, 128),
                        nb_val_samples=1000,
                        nb_epoch=1)
        cnt +=1
    

    output_path = "%s/%s"%(conf.model_dir, conf.model)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model.save("%s/model.h5"%(output_path))

    print ("Model saved")
    