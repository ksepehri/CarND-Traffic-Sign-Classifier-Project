#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-media/samples-chart.png "samples"
[image2]: ./writeup-media/class-10.png "class 10 normal"
[image3]: ./writeup-media/class-10-grayscale.png "class 10 grayscale"
[image4]: ./traffic-sign-test-images/e-25.jpg "Traffic Sign 1"
[image5]: ./traffic-sign-test-images/d-32.jpg "Traffic Sign 2"
[image6]: ./traffic-sign-test-images/b-22.jpg "Traffic Sign 3"
[image7]: ./traffic-sign-test-images/e-40.jpg "Traffic Sign 4"
[image8]: ./traffic-sign-test-images/c-14.jpg "Traffic Sign 5"
[image9]: ./traffic-sign-test-images/a-11.jpg "Traffic Sign 6"
[image10]: ./writeup-media/class-10-he-rotate.png "class 10 histogram equalization rotate"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ksepehri/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed per class for the training set, note that some classes have a lot more data than others. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it creates better contrast especially on dark images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

As a last step, I normalized the image data because we always want our variables to have zero mean and equal variance whenever possible. This is so the optimizer has to do less searching to find a good solution.

After some testing of the model, I improved the processing pipeline by performing histogram equalization and augmenting the data through rotation.

![alt text][image10]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution       	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6	|
| Convolution       	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten				| outputs 400  									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet-5 implementation with RELU as an activation function. I experimented with different batch sizes, epochs, and learning rate. The values that worked best for me were

Learning Rate: 0.001
Batches: 128
Epochs: 100

The number of epochs was the key to hit the 93% accuracy.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .937
* validation set accuracy of 1.00
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    LeNet-5. It was chosen because I'd already implemented it in the previous lab and wanted to try it against a large dataset.
* What were some problems with the initial architecture?
    It was not accurate enough. I had to preprocess the data and normalize it correctly in order to hit the minimum.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    The architecture was adjusted through converting to grayscale, normalizing. The layers were not changed as the preprocessing resulted in good fit of the data.
* Which parameters were tuned? How were they adjusted and why?
    I tried changing all the parameters and testing which made the largest difference in accuracy. In the end Epochs seemed most important to accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Convolution layers worked well with this problem due to less weights being shared across space. At the end of a convolution only the parameters that relate to content remain, which helps with image classification.

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet-5
* Why did you believe it would be relevant to the traffic sign application?
    It's good for image classification and has been around for a long time so it's been vetted.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web:

![alt text][image4]
25 - Road work - ~1400 samples, clear image centered and good lighting, should have good chance.
![alt text][image5]
32 - End of all speed and passing limits - only ~200 samples. sign at an angle, will probably have a difficult time classifying.
![alt text][image6] 
22 - Bumpy road - only ~300 samples, sign at an angle and a few have the same shape, difficult to classify.
![alt text][image7]
40 - Roundabout mandatory - only ~300 samples. good angle of sign and arrows clear to see, reasonable chance of classifying.
![alt text][image8]
14 - Stop - there are about ~750 samples of this image, it should be easier to identify due to its distinct shape and quality/angle of this image.
![alt text][image9]
11 - Right-of-way at the next intersection - around ~1200 samples, clear sign good angle it should classify it easily.

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction, without histogram equalization and rotation:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work								| Road work   									| 
| End of all speed and passing limits	| Keep right									|
| Bumpy road							| Bicycles crossing								|
| Roundabout mandatory					| Roundabout mandatory			 				|
| Stop									| Stop      									|
| Right-of-way at the next intersection	| Beware of ice/snow							|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%.

Here are the results of the prediction, WITH histogram equalization and rotation:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work								| Road work   									| 
| End of all speed and passing limits	| Keep right									|
| Bumpy road							| Bumpy road								|
| Roundabout mandatory					| No passing			 				|
| Stop									| Stop      									|
| Right-of-way at the next intersection	| Right-of-way at the next intersection							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

These are results of the 2nd iteration with includes histogram equalization and rotation. Overall the model seems to be overfitting less than before.

For the first image, the model is very sure that this is a road work sign (probability of 1), and the image does contain a road work sign. There are 480 samples in the test set with an accuracy of 0.91 so this model is fitting ok. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Road work   									| 


For the 2nd image, the model is very sure that this is a keep right sign (probability of .99), and the image is NOT a keep right sign. There are 60 samples in the test set with an accuracy of 0.817 so this model is overfitting. The top soft max probability was

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep right   									| 

For the 3nd image, the model is very sure that this is a Bumpy road sign (probability of 0.99), and the image is a Bumpy road sign. There are 120 samples in the test set with an accuracy of 0.883 so this model is fitting ok. The top soft max probability was

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Bicycles crossing								| 

For the 4th image, the model is very sure that this is a No passing sign (probability of 1), and the image is NOT a No passing sign. There are 90 samples in the test set with an accuracy of 0.778 so this model is overfitting. The top soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| 1          			| No passing         							| 

For the 5th image, the model is very sure that this is a Stop sign (probability of 0.99), and the image does contain a Stop sign. There are 270 samples in the test set with an accuracy of 0.948 so this model is fitting ok. The top soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop											|

For the 6th image, the model is somewhat sure that this is a Right-of-way at the next intersection (probability of 0.74), and the image is a Right-of-way at the next intersection. There are 420 samples in the test set with an accuracy of 0.967 so this model is underfitting. The top soft max probability was

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .745         			| Right-of-way at the next intersection			|
| .255        			| Beware of ice/snow							|


