# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:19:30 2018

@author: pierr
"""
import os

os.chdir('C:\\Users\pierr\Desktop\Data Science - Astronomy\Decision Trees')

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  # Set up features array
  features = np.zeros(shape=(len(data), 4))
  
  # Compute features from data
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  
  # set up target
  targets = data['redshift']

  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))


# Complete the following function
def accuracy_by_treedepth(features, targets, depths):
  # split the data into testing and training sets
  split = 2*features.shape[0]//3
  train_features, test_features = features[:split], features[split:]
  train_targets, test_targets = targets[:split], targets[split:]

  # initialise arrays or lists to store the accuracies for the below loop
  train_diffs = []
  test_diffs = []
  
  # loop through depths
  for depth in depths:
    # initialize model with the maximum depth. 
    dtr = DecisionTreeRegressor(max_depth=depth)

    # train the model using the training set
    dtr.fit(train_features, train_targets)
    
    # get the predictions for the training set and calculate their median_diff
    predictions = dtr.predict(train_features)
    train_diffs.append(median_diff(train_targets, predictions))
    
    # get the predictions for the testing set and calculate their median_diff
    predictions = dtr.predict(test_features)
    test_diffs.append(median_diff(test_targets, predictions))
    
  # return the accuracies for the training and testing sets
  return train_diffs, test_diffs

if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  # Generate several depths to test
  tree_depths = [i for i in range(1, 36, 2)]

  # Call the function
  train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
  print("Depth with lowest median difference : {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))
    
  # Plot the results
  train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
  test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
  plt.xlabel("Maximum Tree Depth")
  plt.ylabel("Median of Differences")
  plt.legend()
  plt.show()
