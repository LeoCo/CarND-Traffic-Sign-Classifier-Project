#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[img1]: ./images/train_dist.png "Train Samples Distribution"
[img2]: ./images/valid_dist.png "Validation Samples Distribution"
[img3]: ./images/test_dist.png "Test Samples Distribution"
[img4]: ./images/standard.png "Standard"
[img5]: ./images/modified.png "Modified"
[img6]: ./test-signs/sign1.jpg "Test Sign 1"
[img7]: ./test-signs/sign2.jpg "Test Sign 2"
[img8]: ./test-signs/sign3.jpg "Test Sign 3"
[img9]: ./test-signs/sign4.jpg "Test Sign 4"
[img10]: ./test-signs/sign5.jpg "Test Sign 5"
[img11]: ./images/softmax.png "Softmax"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/LeoCo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart for each set (train, validation and test) showing how the data is distributed. It is possible to see that some signs are represented with many samples while others are under represented. This is one of the reasons why in the following part the dataset was augmented.

![alt text][img1]
![alt text][img2]
![alt text][img3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To start I decided to generate additional data because the training set was not big enough to generalize 43 different kind of traffic signs.

To add more data in the training set I decided to implement a function that could translate, rotate and/or scale randomly the original image.

Here is an example of the image before and after a random transformation:
![alt text][img4]
![alt text][img5]

The original training set has 34,799 signs while the new augmented set has 173,995.

After that, I decided to convert the images to grayscale because the neural net was performing much better with gray images. A color image has 3 dimensions (RGB) while a grey image has just 1 dimension, so having colors makes the inputs of the neural net more complex by a factor of 3. Unfortunately we have a small training set to train color images for 43 different labels.

The new gray images have dimensions of 32x32x1 instead of 32x32x3.

As a last step, I normalized the image data and shuffle the training set.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As a starting model I used a LeNet neural network. Unfortunately LeNet was created to predict 10 labels and it was not enough accurate on our dataset. That is why I decided to make LeNet deeper and use dropout to regularize our new deeper neural net.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3  	| 1x1 stride, valid padding, outputs 28x28x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x100 				|
| Dropout	    |      				|
| Convolution 3x3  	| 1x1 stride, valid padding, outputs 10x10x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x100 				|
| Dropout	    |      				|
| Flatten	    | 		outputs 2500	|
| Fully connected		| outputs 400 |
| Fully connected		| outputs 150 |
| Fully connected		| outputs 43 |
| Softmax				| 		|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.0003 for 10 epochs splitting the training set in batches of 128 inputs. To regularize the training I have used a dropout factor of 0.75.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy fluctuating aroung 0.965-0.973
* test set accuracy of 0.967

If an iterative approach was chosen:

* I used a LeNet architecture as a starting point
* I adapted LeNet to receive as an input a 32x32x3 color image
* The test set accuracy was 0.73
* Then I tried to convert the image to gray scale and changed LeNet to accept a 32x32x1 gray image
* The test set accuracy improved to 0.82
* Since the validation set and test set accuracy was similar during each epoch, I understood the neural net was underfitting
* I increased the dimension of the two convolutional layers to 100
* I also increased thd dimension of the fully connected layers
* In order to regularize the new more complex neural net I implemented two dropout layers after the max pooling layers
* The new test set accuracy was of 0.92
* The neural net was overfitting (the validation accuracy was much greater than the test accuracy) so I decided to augment the training set
* After augmenting the training set I reached a test set accuracy of 0.967

The final neural net was tuned lowering the learning rate and chosing an appropriate dropout ratio.

LeNet was choosed as a starting neural net because is a solid architecture to classify images, unfortunately LeNet was designed to predict 10 labels while in our dataset we have 43 labels. This is why I needed to make the neural net more complex and deeper.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][img6]
![alt text][img7]
![alt text][img8] 
![alt text][img9]
![alt text][img10]

The first image might be difficult to classify because the yield sign is not in the center and it is surrounded by other shapes (squares, rectangles) that the neural net could wrongly predict.

The second image was choosen to because the 50 km/h sign is similar to the 80 km/h sign and I wanted to see the ability to distinguish between numbers.

The other images should be easy to predict.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield Sign      		| Ahead Only   									| 
| 50 km/h    			| 80 km/h 										|
| Keep Left					| Keep Left											|
| Bumpy Road	      		| Bumpy Road					 				|
| Stop Sign			| Stop Sign      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

This is how expected because the first two signs were choosen to test the prediction ability in difficult scenarios.

Since the test set is made of picture very similar to the last three, this compares favorably to the accuracy on the test set.

In any case this test highlights that the training set should be bigger to better generalize real world scenarios.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

A better understanding of the prediction can be inferred using the following chart:

![alt text][img11]

It is easy to see that for the first image the neural net fails to get even close to the real prediction, in fact the Yield Sign is just in the 5th place. The neural net in not powerful enough to predict not centered images that have noisy shapes around the sign.

The second image shows clearly that the neural net is not powerful enough to distinguish correctly the 5 from the 8 in the signs. This could be because the net recognise other shapes but is not deep enough to learn the difference 5 from the 8. Using another convolutional layer or making the Neural Net deeper could improve it.

The last three images were perfectly predicted and the neural net worked as expected.