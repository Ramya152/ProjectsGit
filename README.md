# Dental Image Classification using Convolutional Neural Networks

This project utilizes Convolutional Neural Networks (CNNs) to classify dental images into two categories: periodontal(periodontitis: an infection of the tissues that hold your teeth in place) and non-periodontal. The dataset consists of dental images collected from kaggle.
Link to the dataset: https://www.kaggle.com/datasets/hasnitadita/image-dental-panoramic/data

## Overview

The classification model is implemented using TensorFlow and Keras. The architecture of the CNN model comprises several convolutional layers followed by max-pooling layers. The final layer uses a sigmoid activation function to output binary predictions.

## Dataset

The dataset is organized into three main directories:
- `training`: Contains training images for model training.
- `validation`: Contains validation images for model evaluation during training.
- `test`: Contains unseen test images for final evaluation of the trained model.

Each directory is further divided into subdirectories representing the two classes: `periodontal` and `non-periodontal`.

## Model Training

- The model is trained using the `train_generator` data generator, which applies various data augmentation techniques such as rescaling, shear range, zoom range, horizontal and vertical flips, and rotation range.
- The training process is monitored using both training and validation data to track the loss and accuracy metrics across epochs.

## Model Evaluation

- After training, the model is evaluated using the `test_generator` data generator on unseen test images.
- Evaluation metrics such as loss and accuracy are calculated to assess the performance of the trained model on the test dataset.

## Results

- The model achieved an accuracy rate of X% on the test dataset.
- Loss and accuracy curves during training and validation are visualized using matplotlib.

## Prediction on New Images

- The trained model can be used to make predictions on new images.
- An example is provided to load and preprocess a single image and obtain a prediction using the trained model.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- Pandas
