# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_traindata_dist.png "traindata_distribution"
[image2]: ./examples/y_label_dataset.png "labels in train data"
[image3]: ./examples/sign_original_gray_normalized.png "image before and after processing"
[image4]: ./sign_from_web/sign_0.png "Traffic Sign 0"
[image5]: ./sign_from_web/sign_1.png "Traffic Sign 1"
[image6]: ./sign_from_web/sign_2.png "Traffic Sign 2"
[image7]: ./sign_from_web/sign_3.png "Traffic Sign 3"
[image8]: ./sign_from_web/sign_4.png "Traffic Sign 4"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

A quick analysis shows the distribution of the classes in training data. From the bar plot, some class has many images while some has a few. 

![alt text][image1]
And the following plot shows how labels exist in the y_train dataset. many data of the same class are saved consecutively, which indicates the shuffle is necessary before training the data.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


To train the data, a pre-processing is required. As the signs image are taken in various conditions: with/without good sunshine, too much/too little exposure, the grayscale image is better for training. I first run an average of 3 channels for each image and the training accuracy is not that good. Changing to real grayscale images significantly improves the accuracy.

After converting the image to grayscale, I normalized the image data to make all the data with a mean around 0 and ranges from -1 to 1. This helps the SGD.

Here is an example of a traffic sign image: original image / normalized image / grayscale image
here I am using opencv to do color conversion. 

![alt text][image3]

All the color image will be converted to gray image(32x32x1) and be used in the training

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I decided to use LeNet which including:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten				|	convert to 1x1x400									|
| Fully connected		| from 400 to 120     									|
| Dropout				|	keep_prob 0.5										|
| Fully connected		| from 400 to 120     									|
| RELU					|												|
| Dropout				|	keep_prob 0.5										|
| Fully connected		| from 120 to 84     									|
| RELU					|												|
| Dropout				|	keep_prob 0.5										|
| Fully connected		| from 84 to 43     									|
| logits				| return      									|
| softmax&entropy		|       									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the epoch is set to 50 and batch_size is set to 128. This setting seems to be good. The learning rate is set to a fine 0.0008. The cross_entropy is calculated by **tf.nn.softmax_cross_entropy_with_logits**. The optimizer is AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.962
* test set accuracy of 0.93
It turns out current architecture works well on validation set.

To get this results, what I did were:
* increase epoch from 20 to 50
* using finer learning_rate from 0.001 to 0.0008
* dropout layer also helps

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image may be hard as it is small and vague and I did find another sign that has similar profile with this sign. The last one also looks not clear, but since the shape is quite special, if the training set includes this class, it might not be an issue.
All the other three images are clear patterns or arrows, which should be fine for the model.
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limit 50km/h		| speed limit 50km/h   									| 
| ice and snow     			| right-of-way at next intersection 						|
| Pedestrians					| Pedestrians											|
| go straight or left	      		| go straight or left					 				|
| no passing for vehicles over 3.5 tons			| no passing for vehicles over 3.5tons			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
the model performs quite well on test dataset(0.93) and since test is much more images than just 5 images.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The probability of each prediction: (class_id : probability sign name)
--------image 0----------
2 :1.000 Speed limit (50km/h)
	
1 :0.000 Speed limit (30km/h)
	
3 :0.000 Speed limit (60km/h)
	
5 :0.000 Speed limit (80km/h)
	
7 :0.000 Speed limit (100km/h)

--------image 1----------
11 :0.897 Right-of-way at the next intersection
	
19 :0.057 Dangerous curve to the left
	
21 :0.029 Double curve
	
30 :0.007 Beware of ice/snow
	
40 :0.004 Roundabout mandatory


--------image 2----------
27 :0.551 Pedestrians
	
11 :0.449 Right-of-way at the next intersection
	
18 :0.000 General caution
	
40 :0.000 Roundabout mandatory
	
30 :0.000 Beware of ice/snow


--------image 3----------
37 :1.000 Go straight or left
	
39 :0.000 Keep left
	
33 :0.000 Turn right ahead
	
26 :0.000 Traffic signals
	
18 :0.000 General caution


--------image 4----------
10 :1.000 No passing for vehicles over 3.5 metric tons
	
12 :0.000 Priority road
	
9 :0.000 No passing
	
40 :0.000 Roundabout mandatory
	
17 :0.000 No entry


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


