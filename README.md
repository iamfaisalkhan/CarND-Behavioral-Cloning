#**Behavioral Cloning** 

The  main aim was to train a convolution neurla network to clone human driing behavior using the recently open sourced [Udacity Car Sim] (https://github.com/udacity/self-driving-car-sim). 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/sample_images.svg "random sample of images with varying angles"
[image2]: ./resources/angle_distribution.png "Steering ngle distribution"
[image3]: ./resources/translate_example.svg "Recovery Image"
[image4]: ./resources/left_right_camera.svg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

---
###Code Organization

The project consist of following code files:

* model is the main entry point that creates and trains the model.
* cnn_models.py comprises of different models that we experimented for building the clonning behavior. 
* generator.py has code for generating training and validation data for this project. 
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained convolution neural network 
* README.md The report
* writeup_report.pdf The PDF of this README

In order to train the model with the final network used for the project run the following commands. 

```sh
python model.py --epochs 10 --model nvidia_relu_dropout 
```

The model.py

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Network Architecture

We experimented with multiple models. Take a look at the cnn_models.py for a detailed list of models used during the experimentation.

Our final model is based on a modified version of the Nvidia's End-to-End learning model. The original model did not use any regularization. However, we noticed that adding the dropout on this models can help to generalize the driving between the different (but similar) tracks including the one not seen by the model before. The input to the model is an image of 200 by 66 as used in the original Nvidia's model. We use a normalization layer to normalize the input images. Below is our model definition in Keras (**cnn_models.py:158-182**). The ReLU was used for non-linearity. The experimentation with other activation layers didn't seem to work very well with this model and require further investigation. 

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

## Training

### Input Data

The model was trained using the Udacity's dataset. The dataset contains about 8K immages each of 3 camera positions (left, center, and right). The left and right camera images are useful for correcting the steering angle in case the car wanders off to the road edges. 

Here is a random sample of images at different steering angle. 

![alt text][image1]

To further investigate the data, we also look at the distribution of steering angle (see figure below) andand quickly found out that the data is heavily unblanced towarded the zero steering angle. This was completely expected as our simulated car will be driving striaght. We handled this by using python **generator** that randomly picks images by first selecting images with only non-zero steering angles but later introducing zero-based steering angle in the sampled images. The threshold for rejecting zero-angled images is adjusted at the beginning of each iteration. 

![alt text][image2]

### Augmentation

We generate extra training examples by augmenting the existing one. Such augmentation allow the model to learn a more general representation from the training data. Here is a brief summary of all the augmentation used in building the final model. 

#### Left and Right Camera

Images from both the right and the left camera were used during the training phase to add the path recovery from the edges of the road. A steering correction factor of 0.2 was used. The experimentation with 

![image4]

#### Random Brightness

A random factor of brightness was added to each image using the HSV colorspace. This can potentially allow us to generalize beyond the seen data and handle shadows in the tracks. 

```python
brightness = 0.25 + np.random.uniform()
img[:, :, 2] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2] * brightness
```
#### Translation 

To simulate the car being at different positions of the track, we randomly translate images in the X and y direction along with the appropriate angle adjustment. Here is an example showing original and translated image from our sample data. 

![image3]

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

To facilitate the generation of large number of data. 


In addition to these augmentation, we experimented with few other techniques including image rotation and using difference image but those are still work in progress. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

