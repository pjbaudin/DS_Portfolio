# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:09:47 2018

@author: pierr
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from support_functions import plot_confusion_matrix, generate_features_targets

# Function to split dataset into train and test dataset
def splitdata_train_test(data, fraction_training):
  # shuffle dataset
  np.random.shuffle(data)
  split_index = int(fraction_training*len(data))
  return data[:split_index], data[split_index:]

# Function to generate features from the dataset
def generate_features_targets(data):
  targets = data['class']

  features = np.empty(shape=(len(data), 13))
  features[:, 0] = data['u-g']
  features[:, 1] = data['g-r']
  features[:, 2] = data['r-i']
  features[:, 3] = data['i-z']
  features[:, 4] = data['ecc']
  features[:, 5] = data['m4_u']
  features[:, 6] = data['m4_g']
  features[:, 7] = data['m4_r']
  features[:, 8] = data['m4_i']
  features[:, 9] = data['m4_z']
  # concentration in u filter
  features[:, 10] = data['petroR50_u']/data['petroR90_u']
  # concentration in r filter
  features[:, 11] = data['petroR50_r']/data['petroR90_r']
  # concentration in z filter
  features[:, 12] = data['petroR50_z']/data['petroR90_z']

  return features, targets

# complete this function by splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
  # split the data into training and testing sets using a training fraction of 0.7
  train, test = splitdata_train_test(data, 0.7)
  # generate the feature and targets for the training and test sets
  # i.e. train_features, train_targets, test_features, test_targets
  train_features, train_targets = generate_features(train)
  test_features, test_targets = generate_features(test)

  # instantiate a decision tree classifier
  dtc = DecisionTreeClassifier()
  # train the classifier with the train_features and train_targets
  dtc.fit(train_features, train_targets)
  # get predictions for the test_features
  predictions = dtc.predict(test_features)
  # return the predictions and the test_targets
  return predictions, test_targets

# Implement the following function
def calculate_accuracy(predicted, actual):
  return sum(predicted == actual)/len(actual)


if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # split the data
  features, targets = generate_features_targets(data)

  # train the model to get predicted and actual classes
  dtc = DecisionTreeClassifier()
  predicted = cross_val_predict(dtc, features, targets, cv=10)

  # calculate the model score using your function
  model_score = calculate_accuracy(predicted, targets)
  print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(targets))
  model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()

# Random Forest
# complete this function to get predictions from a random forest classifier
def rf_predict_actual(data, n_estimators):
  # generate the features and targets
  features, targets = generate_features_targets(data)
  # instantiate a random forest classifier using n estimators
  rfc = RandomForestClassifier(n_estimators=n_estimators)
  # get predictions using 10-fold cross validation with cross_val_predict
  predicted = cross_val_predict(rfc, features, target, cv=10)
  # return the predictions and their actual classes
  return predicted, targets