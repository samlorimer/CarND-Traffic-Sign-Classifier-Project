# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./my_examples/visualisation.png "Visualisation"
[image2]: ./my_examples/hard1.png "Hard training image 1"
[image3]: ./my_examples/hard2.png "Hard training image 2"
[image4]: ./my_examples/hard3.png "Hard training image 3"
[image5]: ./images/originals/1.jpg "Internet image 1"
[image6]: ./images/originals/2.jpg "Internet image 2"
[image7]: ./images/originals/3.jpg "Internet image 3"
[image8]: ./images/originals/4.jpg "Internet image 4"
[image9]: ./images/originals/5.jpg "Internet image 5"
[image10]: ./my_examples/signs_resized.png "Signs resized"
[image11]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/samlorimer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

I used a basic random selection of images within the dataset to look through some sample images, along with their classification label.  In this case, 34 translates to "Turn left ahead".

![alt text][image1]

I was quite surprised to see how dark and difficult to decipher some of the images were to the naked eye - included 3 examples of this below where brightness differences make the sign hard to read.  
![alt_text][image2] ![alt_text][image3] ![alt_text][image4]

Surely getting 93% accuracy would be a big challenge with that type of input...  

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The only preprocessing I applied to the images was to normalise them so that all their channel data was between -1 and 1.  Since the original channel values were between 0-255, this was done by subtracting 128 from each pixel, then dividing by 128.

I did this so that I could see what type of accuracy I was able to achieve with minor modifications to the model architecture, and then could come back and apply more preprocessing as required to reach the accuracy level needed.  

As it turned out, no other pre-processing was needed but I intend to revisit this section in future to try out the other techniques that I've seen discussed, such as greyscale, generating additional data for the classes with the least number of included examples in the dataset and other normalisation techniques.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was modified from the LeNet architecture with the addition of dropout, and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU activation		|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120									|
| RELU activation		|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| Outputs 84									|
| RELU activation		|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| Outputs 43 (one per sign class)				|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimiser as a slightly more efficient version of the optimisation used in the course materials.  I left the batch size at 128 as per the LeNet architecture, and left Epochs at 10 (mostly because I was doing the majority of training locally on my laptop during flights and wanted it to run reasonably quickly!).  

A learning rate of 0.001 was used - the course notes suggested it as a sensible default and it provided good results in my testing.

Before training, I also shuffled the training data to ensure the model was not biased toward the order of the images.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.961
* validation set accuracy of 0.930
* test set accuracy of 0.920

The architecture decisions were:
* I based my initial architecture heavily on the LeNet example, as it was already a capable CNN for recognising greyscale handwritten digits in images, and could be easily converted to recognise 3-channel colour traffic signs with minor modifications to the depth of the layers.  I changed the input to be 3 channels, appropriate changes to the depth of the interim layers and then ensuring the final output layer size was 43 to match the number of sign classes.
* I originally saw a validation set accuracy of ~0.860 when using the raw LeNet architecture, and this increased to about 0.875 when normalising the images
* Since this accuracy was not sufficient, I was interested to experiment with adding dropout layers to ensure the network developed many redundant paths and may be better able to deal with some of the more difficult input images
* I first applied a dropout step with a keep probability of 0.5 after every relu activation in training the model.  After ensuring that I was correctly applying a keep probability of 1 to the validation of the model, I seemed to experience more variability in the validation results after this, but they were only improved to ~0.890 in most cases.
* I then experimented with varying the keep probability so that it was slightly higher (0.7) for the CNN layers in case I was losing too much information from the images, and found that this did improve the model to 0.900+
* Finally I removed the dropout steps completely from the CNN layers and only left them on the fully connected layers with a keep probability of 0.5, and increased the validation set results to 0.930 and above.
* The resulting model had a good accuracy on both the training and validation sets.  It is likely still overfitting a little to the training data but my next steps if I wanted to improve the accuracy further would likely be to investigate more preprocessing steps.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found using google image search.

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

I then cropped them to show as much of the sign as possible, and resized to 32x32 pixels with 3 channels (no alpha channel).  The resulting input images looked like this:

![alt_text][image10]

I expected that all of these images should be relatively easy for the model to classify, because they are all taken in daylight conditions with the sign clearly visible.  The stop sign image was cropped such that the sides of the sign are slightly obscured which may make it a little more difficult, but the other features of the sign are evident.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image								|     Prediction	        			| 
|:---------------------------------:|:-------------------------------------:| 
| Yield       						| Yield   								| 
| No vehicles    					| No vehicles 							|
| Stop								| Stop									|
| Right-of-way at next intersection	| Right-of-way at next intersection		|
| Road work							| Road work      						|


The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%, but as mentioned above the images in this new set were all very clearly visible and well-cropped.  When viewed through the limited lens of these 5 clear images, it appears the model is highly accurate in good conditions.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is extremely sure that this is a yield sign (probability very close to 100%), and the image does contain a yield sign - the other probabilities were extremely small.

The model produced similarly definite results for the no vehicles sign, also with extremely close to 100% probability, and had 99+% probability for the right-of-way and road work images as well.

The least certain (but still pretty definite) prediction was the Stop sign (third image) where the edges were slightly clipped in the image.  It had the following top-5 softmax predictions:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .893         			| Stop   										| 
| .074     				| Yield 										|
| .015					| Bumpy road									|
| .009	      			| No vehicles					 				|
| .006				    | No entry      								|

This shows it was still very certain about the prediction as a stop sign, but some of the feature maps in the neural network must be looking for the angled edges and similar red/white colouring of parts of this sign in order to also consider a yield sign as the second most likely.

