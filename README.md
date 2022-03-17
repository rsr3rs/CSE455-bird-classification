# CSE455-bird-classification

## Group Members
Robin Yang, Jack Zhang

## Problem description
For our project, we dicided to go with the recommended bird classification. The task is to identify the species of a bird from its image.

## Dataset and Technologies used
Our dataset is from [kaggle](https://www.kaggle.com/c/birds-22wi/data). The dataset contained more than 500 birds species to classify, with over 20k of images of training data.
Seeing how this is a relatively sophisticated computer vision task, we decided to use a [ResNet18](https://arxiv.org/abs/1512.03385) model. We also used a pretrained model to take advantage of the feature extracting. Although our data have different sizes, CNN uses convolution layers and pooling layers for its main connections, which means we don't have to do any cropping. We downsampled our images to $224 \times 224$ to fit the original ResNet18 input size, and changed the final fully connected layer to have the same amount of class size. This is all implemented with PyTorch.

## Training Methodology
We used cross entropy as the final loss metric for our model, as it fits well for a classification task. For our optimizer, we used stochastic gradient descent with a batch size of 16. We chose an initial learning rate of $5e-2$, and used a scheduler to decrease it exponentially every 7 epochs. The dataset didn't come with any testing labels, so we divided our dataset into training and validation sets for evaluating the model mid-training. We trained our model on [Google Colab](https://colab.research.google.com/) using their GPUs. Unfortunately, the training time exceeded the personal use limit for Google Colab, so we saved our model checkpoints and trained on several accounts. We trained for a total of around 4 days, and trained for over 80 epochs in total.

## Training Result and Model Evaluation
