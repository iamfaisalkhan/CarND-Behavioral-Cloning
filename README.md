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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
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

Our final model is based on the Nvidia's End-to-End learning model. The original model doesn't use any regularization. However, we noticed that adding the dropout on these models can help to generalize the driving behavior between the different (but similar) simulated tracks even on the track not seen by the model before. 

```python
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
```

We followed the convolution neural network My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 


###. Training

The model was trained using the Udacity's dataset. The dataset contains about 8K immages each from 3 camera positions (left, center, and right). The left and right camera images are useful for correcting the wandering off behavior of the simulated car. 

TODO: Sample Images
TODO: a sample of 

###. Augmentation

Given the limited amount of the data from


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

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
