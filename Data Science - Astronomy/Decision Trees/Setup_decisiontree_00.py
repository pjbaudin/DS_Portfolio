# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:16:20 2018

@author: pierr
"""
import os
import numpy as np

os.chdir('C:\\Users\pierr\Desktop\Data Science - Astronomy\Decision Trees')

data = np.load('sdss_galaxy_colors.npy')

import numpy as np

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

    
# call our function 
features, targets = get_features_targets(data)
    
# print the shape of the returned arrays
print(features[:2])
print(targets[:2])

# Setup decision trees 
from sklearn.tree import DecisionTreeRegressor

# initialize model
dtr = DecisionTreeRegressor()
# train the model
dtr.fit(features, targets)

# make predictions using the same features
predictions = dtr.predict(features)

# print out the first 4 predicted redshifts
print(predictions[:4])


# write a function that calculates the median of the differences
# between our predicted and actual values
def median_diff(predicted, actual):
  return np.median(np.abs(predicted[:] - actual[:]))

# call your function to measure the accuracy of the predictions
diff = median_diff(predictions, targets)

# print the median difference
print("Median difference: {:0.6f}".format(diff))

