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
        '--samples',
        type=int,
        nargs='?',
        default=conf.samples_per_epoch,
        help='Path to image folder. This is where the trained model will be saved.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        nargs='?',
        default=conf.batch_size,
        help='Path to image folder. This is where the trained model will be saved.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default=conf.data_folder,
        help='Path to image folder.'
    )
    parser.add_argument(
        'model_dir',
        type=str,
        nargs='?',
        default=conf.model_dir,
        help='Path to image folder. This is where the trained model will be saved.'
    )

    args = parser.parse_args()
    
    conf.epochs = args.epochs
    conf.data_folder = args.image_folder
    conf.epochs = args.epochs
    conf.model_dir = args.model_dir
    conf.model = args.model
    conf.samples_per_epoch = args.samples
    conf.batch_size = args.batch_size

    data = pd.read_csv("%s/driving_log.csv"%conf.data_folder)
    # mask = np.random.rand(data.shape[0]) < 0.9
    # train = data[mask] 
    # valid = data[~mask]

    # Pick model
    if conf.model == "nvidia1":
        model = model_nivida1()
    elif conf.model == 'nvidia2':
        model = model_nivida2()
    elif conf.model == 'nvidia2b':
        model = model_nivida2b()
    elif conf.model == 'nvidia3':
        model = model_nvidia3()
    elif conf.model == 'nvidia_relu':        
        model = model_nvidia_relu()
    elif conf.model == 'nvidia_relu_dropout':
        model = model_nvidia_relu_dropout()
    elif conf.model == 'nvidia_elu':
        model = model_nvidia_elu()
    elif conf.model == 'comma_elu':
        model = model_comma_elu()
    elif conf.model == 'comma_relu':
        model = model_comma_relu()
    elif conf.model == 'comma_lrelu':
        model = model_comma_lrelu()
    else:
        print("Model not defined")
        exit()
    
    if args.model_init != None and os.path.exists(args.model_init):
        print ("Model re-loaded from %s"%(args.model_init))
        model = load_model(args.model_init)

    model.summary()

    cnt = 0

    output_path = "%s/%s"%(conf.model_dir, conf.model)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    val_loss = 100.0
    for i in range(conf.epochs):
        bias = 1. / (cnt + 1.)
    
        history = model.fit_generator(
                        training_generator(data, bias, conf.batch_size), 
                        samples_per_epoch=conf.samples_per_epoch, 
                        validation_data=validation_generator(data, conf.batch_size),
                        nb_val_samples=1000,
                        nb_epoch=1)
        if (history.history['val_loss'][0] < val_loss):
            val_loss = history.history['val_loss'][0]
    
        model_name = "%d_%s_%.4f.h5"%((cnt+1), conf.model, history.history['val_loss'][0])
        model.save("%s/%s"%(output_path, model_name))

        cnt +=1

    print("Best loss %.4f"%val_loss)