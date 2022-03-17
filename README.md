# CSE455bird classification

## Project description
We are training a neural network to classify different types of birds using 
a bird [dataset](https://www.kaggle.com/c/birds-22wi/overview).

The neural network we chose was ResNet18 which consists of 18 convolutional
layers and residual features. We used a pretrained model that is capable of
classifying an image of size 224*224 into 1000 categories.

Since the dataset is missing test data, we randomly selected 80% of the training
data as the actual training data and the result 20% as the testing data, and
use the same data items through the training process.

After 64 valid epoch, we are able to get a training accuracy of 87.8% and
a test accuracy of 64.9%

## Code used
We adapted code from the transfer learning [tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
and use it on Google Colab. We rewrote the data processing, model training and saving
process to build a pipeline from downloading data to training. It saves all models
on Google Drive for future use. Since we used a pretrained model, we only need
to change the last layer without defining the whole network. We did try some other
models like MobileNetV3, but we decided to use ResNet18 as training is affordable
and it has good performance.

## Some problems
Google Colab disallows us to train continuously, so the training process was run a
few times. Dataset was split randomly so each time we get a different train and test dataset.
We then fixed the issue using a fixed random seed. Another problem was that we were
not saving optimizer and scheduler initially, but we realized the problem after
getting a low accuracy after loading a saved model. Because of the issues, the first
22 epochs were discarded.


 