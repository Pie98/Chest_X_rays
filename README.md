# Chest X-rays image classification

## Overview

This machine learning project, primarily developed using TensorFlow, aims to classify various lung diseases based on radiographic images.

## Table of Contents

- [Project Structure](#Project-Structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#Results)


## Project Structure

The repository is organized as follows:

### 1_Datasets (not present in this repository)

- The starting dataset is the NIH Chest X-rays datset from Kaggle with over 112,000 Chest X-ray images from more than 30.000 unique patients. The dataset we are going to use is a subset of this dataset, obtained preprocessing more than 28.000 images.

### 2_preprocessing

- **Creating_train_test_directories:** In this notebook we select the images we want to use in the project and create the train/valid/test folders. Each folder is then divided in 7 subfolders corresponding to the classes we want to predict ('Atelectasis','Effusion','Infiltration','Mass','No_finding','Nodule','Pneumothorax')

- **Creating_augmented_train_data:** Here we add to the training data augmented images to prevent class imbalances,

## Model Training

Our goal is to create a model with higher accuracy than betting websites predictions. To achieve this, we initially create three different models: a Dense model, a Bidirectional LSTM model, and a CONV1D model. The best-performing models result from experiments and parameter testing, details of which are not documented in the notebook. Lastly, we built a model that, in addition to the statistics from previous matches, incorporates bookmakers' statistics to obtain more accurate predictions.

## Evaluation

In addition to calculating the accuracies for the training, validation, and test datasets, we evaluate how much money would be won or lost by betting on the matches predicted by our best-performing model. Furthermore, another statistic called "money_won" is created, derived from the product of the win probability and the odds, to calculate the expected value of money won in this specific bet.

## Results

Our model performs slightly better than the bookmakers one. It isn't sufficient to win money sistematically. 
The main limitations of this model are:

- This model incorporates the odds we aim to enhance as a feature. Consequently, our probability of winning is influenced by the bookmakers' odds, making it more challenging to identify favorable odds for potential monetary gains.

- This model is limited by the absence of some important features, such as the number of injured players, information about the coach and the players, the referee, and the outcomes of competitions like Europa League, Champions League, Coppa Italia, etc., which were not available during the data collection process.

- We could try to improve the model by adding some extra features as: number of matches won in the last X matches, number of matches won at home/away in the last X matches, numer of games won against the opponent teams in the last X matches. 
