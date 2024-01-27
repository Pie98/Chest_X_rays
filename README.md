# Chest X-rays image classification

## Overview

This machine learning project, primarily developed using TensorFlow, aims to classify various lung diseases based on radiographic images.

## Table of Contents

- [EDA and preprocessing](#EDA_and_preprocessing)
- [Model Training](#model-training)
- [Results](#Results)


The repository is organized as follows:

## EDA and preprocessing

### 1_Datasets (not present in this repository)

- The source dataset is the NIH Chest X-rays datset from Kaggle, with over 112.000 Chest X-ray images from more than 30.000 unique patients. The dataset we are going to use is a subset of this dataset, obtained preprocessing more than 28.000 images.

### 2_preprocessing

- **Creating_train_test_directories:** In this notebook we select the images we want to use in the project and create the train/valid/test folders. Each folder is then divided in 7 subfolders corresponding to the classes we want to predict ('Atelectasis','Effusion','Infiltration','Mass','No_finding','Nodule','Pneumothorax').

- **Creating_augmented_train_data:** Here we add augmented images to the training dataset to prevent class imbalance.

### 3_EDA

- **Exploratory_data_analysis:** In this notebook we visualize some images, their pixel distribution and the correlation between age or sex and the various diseases.

## Model Training

### 4_models

- **Dense_model:** The first attempt of creating a baseline. The dense model is too simple to learn pattern in this complex dataset.

- **CNN_model:** An improvement from the dense model obtained using a convolutional neural network with average pooling and dropout to reduce overfitting. 

- **Feature_extraction_model:** Before using fine tuning we test how different pretrained models perform with feature extraction. (Here only the EfficientNetB6 model is shown). 

-**Fine_tuning_model** In these four notebooks are trained four different fine tuned models (EfficientNetB7 and EfficientNetB6).
In order to reduce overfitting, on top of Global average pooling layers, we used data augmentation layers. The best performances were obtained setting all the layers to trainable. 

- **Ensemble_model:** Here the 3 best performing models are concatenated to obtain the final model. 

## Results

### 5_results

-**results_evaluation:** In this notebook we evaluate the results of the best performing model (the ensemble_model).

