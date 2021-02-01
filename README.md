<h1 align='center'>FACIAL EXPRESSION AND KEYPOINTS DETECTOR</h1> 

Artificial intelligence is impacting the future of virtually every industry and every human being. Artificial intelligence has acted as the main driver of emerging technologies like big data, robotics and IoT, and it will continue to act as a technological innovator for the foreseeable future. One of the areas where AI will innovate is Computer Vision. In this project I have built Deep Learning Models which will detect and classify the expression and the keypoints on our faces. Apple's FaceID uses a more advanced model to detect the keypoints on a face to give them access to potentially sensitive information. This technology is the future but is application even today. For this project I have built two models which in conjunction will detect both the facial feature keypoints and the emotion.

There are two datasets being used to separately train and test the facial expression classifier and the facial keypoint classifier. This project will three subsections. The first one is for the Keypoint Classifier, the second for the Expression Classifier and the third will be about the combination of both these classifiers. The datasets can be found at the following links: https://www.kaggle.com/c/facial-keypoints-detection/overview and https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge.

## Technologies Used

<details>
<a name="Technologies Used"></a>
<summary>Show/Hide</summary>
<br>

* Python
* Pandas
* Numpy
* Seaborn
* Matplotlib
* CV2
* Tensorflow 2.0
* Keras
* Sci-kit Learn
* Google Collab
</details>

## Facial Keypoint Classifier

### All About the Pictures

After mounting the drive in Google Collab notebook, I just took a quick look at the data for the Facial Keypoint Classifier. The dataset contains 2,140 non-null rows and 30 columns. Column 1 all the way through 29 contain the x and y coordinate values of the keypoints of the images. The last column contains the images pixel values in a space separated string. So the first order of business is to convert the image values into a proper format, like a 2-D numpy array.

**PIC of data values**

Next, I plotted the image with the keypoints overlaid on top just to get a sense of how the keypoints look on a face. The x and y coordinates for each feature are in adjacent columns, that means all the x-coordinates are in even numbered columns while all the y-coordinates are in odd numbered columns. Below is a small example of the images with their keypoints overlaid on top.

**PIC OF GRId pics face**

For an accurate model, we require data. While 2,140 images may seem like a lot, it really isn't. Also for a model to be robust, we have to feed it images which have slight distortions or taken at an angle, so that it can generalize better. To achieve this, we perform image augmentations like randomly flipping, zooming in, changing pixel values of the images in the dataset. Below is a small example:

**Normal dude pic**

**Horizontal flip**

**vertical flip**

**other augs**

After doing all that we have a dataset which contains 8,560 images. Now I normalize the image data by dividing each pixel value of each image with 255. Since all images are in grayscale, the pixel values range only from 0 to 255 with a single colour channel. By dividing with 255, we lower the values of each pixel to anywhere between 0 and 1, while retaining the relational information required for an image to be an image. This lets the model run faster and more efficiently. I then split the data in a train-test split, with 20% of images going to the test data.

### Modelling the Deep Learning Model

I experimented a lot with different deep learning models and quickly found out that a simple dense neural network with only ANNs or CNNs was nowhere enough to achieve the desired result, I suspect because of the vanishing gradient problem. So I started searching online and found out about Residual Networks and their **skip connection** feature which is useful in combating the vanishing gradient problem. After spending a lot of time learning about Residual networks, I decided to use ResNets which include **identity mapping**. I took inspiration from the various Residual Networks and built my own flavour of it, which includes dropout layers, pooling layers and traditional CNNs too. Below shows my design for the Convolutional and Identity Block in the Residual block.

**PIC of Conv block**
**PIC OF identity block**

### Model Compilation and Assessment

I have used the **Adam** optimizer, as it is the best optimizer as it is dynamic. I utilized cross validation with a validation split of 5% to check for any sort of overfitting by using the EarlyStopping callback from Keras & Tensorflow. After training the model multiple times, while changing the architecture of the model multiple times, I got a model which gave me 84% accuracy in detecting the keypoints.

**Pic of accuracy**

## Facial Expression Detection

### About The Data

Now I will build a DL model which detects and classifies emotions from the image. The dataset for this model contains only 2 columns but 24,568 images. The first row contains integers from 0 to 4. Each of these integers stands for an emotion. 0=Angry, 1=Disgust, 2=Sad, 3=Happy, 4=Surprise. The other column contains the image pixel values as a space separated string. Thus, we need to change the format of the image pixel values from a string to a 2-D numpy array.

**PIC OF dataset**

### Data Preparation and Image Augmentation

Just like for the keypoint classifier it is always a good idea to increase the dataset by performing image augmentations to make the model more robust. This makes the model generalize to new unseen data better. I perform horizontal and vertical flipping, zoom, image brightening and rotation using the ImageDataGenerator module from **Tensorflow**. Since this is a multi-class classification problem, it is always a good idea to check how evenly sampled the classes are. As we can see in the bar graph below, there are very few images for the 'Disgusted' expression, while there are more than 6 times the images for emotion 'Happy'. This poses a problem when it comes to classify a disgusted image.

**PIC of the bar**

Next it is crucial to encode the expression category properly. The model does tries to infer relation between integer numbers, while in this case there is no such thing. So I use the One-Hot encoding module from **Sci-kit Learn**. I also do image normalization by dividing the image pixel values by 255. Then I split the data into train-validation-test sets of 45%,45% and 10% respectively. I then use the ImageDataGenerator module to feed in the images in batch size of 64.

### Model Compilation and Assessment

I used the same neural network model for this classifier as well as the keypoint detector, along with an Adam optimizer and EarlyStopping module. The accuracy of the model on the test data which was set aside is 87%. This is very accurate. Below is the confusion matrix heatmap and the classification report of the model performance.

**PICS OF CLASS REPORT N HEAT**

Below we can see the models performance on a random set of 15 images. As expected, whenever there is a picture of a disgusted person, the model performed very poorly. This can easily be remedied by balancing the data better for each class.

**PICS OF 15 faces**

## Combining Both Models

Now that both our classifiers are ready and trained well, we can combine these two models so that they can feed into each other and produce a single output which predicts the keypoint's x and y coordinates and the emotion of the picture. We can assume that with the lower accuracy of 84%, this combination of models is overall accurate by 84%. Below is a plot of a set of images which have been tested after passing through both the models. I say they are pretty accurate.

**PICS of combined pics**

## Further Improvements

While this is a very useful feature in detecting and recognizing the emotion of a human being, the concept of this project unlocks its true potential as a video application rather than on static images. But that requires a lot of processing power and specialized neural networks.
