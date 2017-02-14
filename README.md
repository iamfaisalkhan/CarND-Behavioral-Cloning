#**Behavioral Cloning** 

The  main aim of the behavioral clonning project was to train a convolution neural network to mimic human driving behavior using the open source [Udacity Car Simulator] (https://github.com/udacity/self-driving-car-sim). This document explains our approach to the problem along with the explanation of the training data and final model. 

[//]: # (Image References)

[image1]: ./resources/sample_images.png "random sample of images with varying angles"
[image2]: ./resources/angle_distribution.png "Steering ngle distribution"
[image3]: ./resources/translate_example.png "Recovery Image"
[image4]: ./resources/left_right_camera.png "Recovery Image"
[image5]: ./resources/angle_dist_balanced.png "Angle distrubition balanced"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
##Code Organization

The project is written in python and organized in following files.

* **model.py** is the main entry point that creates and trains the model.
* **cnn_models.py** comprises of different models that we experimented for building the clonning behavior.  The final model is the fuction *model_nvidia_relu_dropout*.
* **generator.py** has code for generating training and validation data for this project. 
* **preprocess.py** has utility functions for augmenting data. 
* **drive.py** is from the simulator for driving the car in autonomous mode
* **model.h5** contains a trained convolution neural network 
* **README.md** The report file

In order to train the model using the final network used in this project run the following commands. 

```sh
python model.py
```

The above command expects to have a data folder under the current directory containg the **IMG** directory and a csv file for steering commands: **driving_log.csv** file. Most of these options are configurable as command line option.

```sh
python model.py -h

usage: model.py [-h] [--epochs [EPOCHS]] [--model [MODEL]]
                [--init [MODEL_INIT]] [--batch_size [BATCH_SIZE]]
                [--image_folder [IMAGE_FOLDER]] [--model_dir [MODEL_DIR]]

Behavior Clonning

optional arguments:
  -h, --help            show this help message and exit
  --epochs [EPOCHS]     Number of training iterations or epochs
  --model [MODEL]       Number of training iterations or epochs
  --init [MODEL_INIT]   Initialize the model with weights from the file.
  --batch_size [BATCH_SIZE]
                        Batch size
  --image_folder [IMAGE_FOLDER]
                        The path to the top level directory contain IMG
                        folder. The top level directory should contain
                        driving_log.csv file
  --model_dir [MODEL_DIR]
                        Output directory for storing the model.

```

In order to run a model to drive the car in the simulator. The trained model *model.h5* is provided along with the code. 

```sh
python train model.h5
```

## Design Approach

The rest of this document explains different part of the projects in some detail. In this section, we go over some high level design decision made to come up with the solution of this problem. 

Our approach  involved experimenting with lot of different models and tunning various parameters. In order to facilitate this, we organized the code in such a way to facilitate experimenting with multiple models and different parameters.  The **cnn_models.py** shows a variation of models mostly the variation on comm.ai or nvidia model with different type of activation, dropout probabilities and normalization layers. 

While training, the output from each epoch was saved. We usually tested the model with lowest validation loss from each model. Testing the model only on Track 1, gave us an idea on how the model is lacking e.g. if it is doing hard right or left turn than adjusting the X and Y translation range values or left/right camera steering offset helped  the next iteration. 

#### Overfitting

We tried to avoid overfitting of the model by:

* Generating lot of augmentation of the input training data. 
* Using a different data generator for training and validation data.  The validation generator only sample images from center camera without angle bias and withnot any further augmentation. 
* We also experimented with training / testing split of the data but the final model uses all the data and rely on augmentation and dropout to avoid overfitting. The performance of the model on Track 2 (see video at the end) is some indicaiton of generalizatin. 

#### Learning Rate

The adam optimizer was used for its adaptable learning rate. 

### Visualization

To begin understanding the data and how the preprocessing is working, used a lot of visulalization (See the two Ipythong notebook for some examples). 

## Network Architecture

We experimented with multiple models. Take a look at the cnn_models.py for a detailed list of models used during the experimentation.

Our final model is based on a modified version of the Nvidia's End-to-End learning model. The original model did not use any regularization. However, we noticed that adding the dropout on this models can help to generalize the driving between the different (but similar) tracks including the one not seen by the model before. The input to the model is an image of 200 by 66 as used in the original Nvidia's model. Additionaly, we used a normalization layer to normalize the input images. Below is our model definition in Keras (**cnn_models.py:158-182**). The ReLU was used for non-linearity. The experimentation with other activation layers (ELU or LeakyELU)  didn't seem to work very well with this data and may require further investigation. 

```python
# cnn_models.py
def model_nvidia_relu_dropout():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape = (conf.row, conf.col, conf.ch)))

    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(36, 5, 5, activation='relu',  subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init='he_normal', border_mode="valid"))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1), init='he_normal',  border_mode="valid"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
```

## Training Data

The model was trained using the Udacity's dataset. The dataset contains about 8K immages each of 3 camera positions (left, center, and right). The left and right camera images are useful for correcting the steering angle in case the car wanders off to the road edges. 

Here is a random sample of images at different steering angle. 

![alt text][image1]

To further investigate the data, we also look at the distribution of steering angle (see figures below) and found out that the data is heavily unblanced towarded the zero steering angle. This was  expected as our simulated car will be driving striaght. We handled this by using python **generator** that bias towards non-zero angles and over the iterations introduce zero-based steering angle images . The threshold for rejecting zero-angled images is adjusted at the beginning of each iteration. 

```python

 threshold = np.random.uniform()
if  abs(y) < 0.1:
     if angle_bias > threshold:
        continue
```

![alt text][image2]

Using the angle bias, we can emphasis the distribution with more non-zero steering angle example vs zero valued angles and vice versa. Here is the comparison distribution plot based on the bias value of 1 (completely ignoring smaller angles) to 0.25 (more balanced distribution)

![alt text][image5]

## Augmentation (Generating More Data)

We generated extra training examples by augmenting the existing one. Such augmentation allow the model to learn a more general representation from the training data. Here is a brief summary of all the augmentation used in building the final model. 

#### Left and Right Camera

Images from both the right and the left camera were used during the training phase to add the path recovery from the edges of the road. A steering correction factor of 0.2 was used. The experimentation with 

![alt text][image4]

#### Random Brightness

A random factor of brightness was added to each image using the HSV colorspace. This can potentially allow us to generalize beyond the seen data and handle shadows and lightning conditions in the tracks. 

```python
brightness = 0.25 + np.random.uniform()
img[:, :, 2] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2] * brightness
```
#### Translation 

To simulate the car being at different positions of the track, we randomly translate images in the X and y direction along with the appropriate angle adjustment. Here is an example showing original and translated image from our sample data. 

  ![alt text][image3]

#### Flip

We randomly flipped images around the y axis to simulate driving in reverse direction. The angle was also reversed for such images. 

```python
    mirror = np.random.randint(2)
    if mirror == 1:
        X = cv2.flip(X, 1)
        y = y * - 1.0
```

#### ROI (Region of Interest)

We cropped top (everything above horizon) and bottom (car hood). Few experiments showed removing top 60 and bottom 20 pixels works the best for this dataset. In future, we might want to compute the horizon and use that as input to the model. 

### Data Generators

To facilitate the generation of large number of training examples without running out of memory, we used python generators along with Keras **fit_generator** function to generate training and validation dataset.  A separate training and validation data generator was used to sample the examples. The validation generator only sample from the center camera examples without any angle bias.  (See generatory.py for more details).

```python
	model.fit_generator(
                        training_generator(data, bias, conf.batch_size), 
                        samples_per_epoch=conf.samples_per_epoch, 
                        validation_data=validation_generator(data, conf.batch_size),
                        nb_val_samples=1000,
                        nb_epoch=1
          )

```

In addition to these augmentation, we experimented with few other techniques including image rotation and using a sequence of difference images but those were not used in the final model. 


### Result

The final model was tested using the simulator at fastest graphics quality and 800x600 resolution. 

**Track 1** | **Track 2**
<a href="http://www.youtube.com/watch?feature=player_embedded&v=bpGDu853EMg
" target="_blank"><img src="http://img.youtube.com/vi/bpGDu853EMg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a> | <a href="http://www.youtube.com/watch?feature=player_embedded&v=bpGDu853EMg
" target="_blank"><img src="http://img.youtube.com/vi/bpGDu853EMg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

### Reflection

The project was  extremly fun and overall a great learning exercise. It also shows the difficult task of building a model to learn to drive in real world. There is a lot of room of improvement in this project. The Udacity's limited data gave us a good model but it is still limited to generalized to the newer track that the simulation team is adding to the simulator. We are also interested in applying some modified version of the winning models from the Udacity's self driving car challenge. The idea of difference image actually came from one of the model (rambo: https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo). 

### Acknowledgements

The project wouldn't have been possible within this time frame if there wasn't for the community's help. I would like to thank the Vivek Yadav and Mohan Karthik for their excellent blog posts that saved me lot of pitfalls early in the project.

