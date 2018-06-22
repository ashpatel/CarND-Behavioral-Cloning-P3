# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[cnn]: ./examples/CNN.png "CNN Architecture"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The provided drive.py was modified to preprocess the camera images before performing the predict function. The preprocess matched the preprocessing applied to the images when training the model, discussed in the training section.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model implemented the convolution neural network described in the Nvidia Automotive paper ([Nvidia Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf))

The visualization of the Nvidia CNN is show in this diagram from the paper;

![alt text][cnn]

#### 2. Attempts to reduce overfitting in the model

With the initial implementation of my CNN, I was getting some overfitting, but the car was still able to traverse the track correctly, though it wasn't a smooth ride. From the visualization of the training it looked like there was a little overfitting occurring.


[train2]: ./examples/training_5.png "training visualization"
![train2]

I added a dropout layer just after the 100 neuron fully connected layer, and that led to less overfitting, and the car was able to traverse the track with a better ride.


[train3]: ./examples/train_dropout.png "training visualization"
![train3]


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I only parameter I tuned was the number of epochs, after adding the dropout layer, I was able to run 5 epochs and get a low loss without overfitting.

#### 4. Appropriate training data

I generated my own test data by employing center driving and doing a few circuits. I augmented this data then by using the two side cameras and adjusting the angle of the steering to account for the view angle. I found a 0.25 degree adjustment worked great, over a 0.2 degree adjustment. For each of these three images, I also generated a flipped version of the image. This would simulate driving the track in reverse and account for the fact that the track only had mainly left hand turns. This greatly increased my training images by 6X. I had about 76K training samples and 19K validation samples.

In addition to my own generated data, I also used the Sample Training Data provided by Udacity. I also processed this training set in the same way by using all three cameras and flipping the images.

This initial set of data was partially successful in navigating the track, but the car would run off the track after the bridge with the sharp turn.
I generated some additional data by driving curved segments of the track and driving off center and then correcting back to center. I did this for some of the trickier corners. This generate a better run, but simulator still had problems at the curve after the bridge, where the siding on the right disappears and it is not able to recognize the edge very well.

I tied pre-processing the images by turning them to Grayscale, but that didn't make much difference. I think remembered using HSV encoding in the first project. So I converted the images to HSV and only used the V channel.
I had to modify the CNN to deal with a (160,320,1) input shape vs (160,320,3).
The image below shows the original image, and the three seperate HSV channels for 5 random training samples.

[HSV]: ./examples/HSV.png "HSV"
![HSV]

I modified the drive.py so that it also pre-processed the images to only use the V channel before it did the prediction. This lead to a better trained model that was able to drive the track successfully. I also turned the car around at the start, and was able to autonomously drive the car in the opposite directly successfully.

The file backward_run.mp4 is a video of autonomous drive of the track in reverse.
