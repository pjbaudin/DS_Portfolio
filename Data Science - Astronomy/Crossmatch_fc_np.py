# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:27:17 2018

@author: pierr
"""

# Write your crossmatch function here.
import numpy as np
import time

# Calculate angular distance between two objects
def angular_dist(r1, d1, r2, d2):
  a = np.sin(np.abs(d1 - d2)/2)**2
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  return 2*np.arcsin(np.sqrt(a + b))

# Crossmatch function upgraded
def crossmatch(dat1, dat2, max_rad):
  # start timer
  start = time.perf_counter()
  # convert maximum distance in radians
  max_rad = np.radians(max_rad)
  
  # Initiate list for match and no match
  matches = []
  no_matches = []
  
  # Convert coordinates to radians
  dat1 = np.radians(dat1)
  dat2 = np.radians(dat2)
  ra2s = dat2[:,0]
  dec2s = dat2[:,1]
  
  # Perform loop for crossmatch
  for id1, (ra1, dec1) in enumerate(dat1):
      # Using numpy to compute min_dist
      dists = angular_dist(ra1, dec1, ra2s, dec2s)
      min_id2 = np.argmin(dists)
      min_dist = dists[min_id2]
      
      # Ignore match if it's outside the maximum radius
      if min_dist > max_rad:
          no_matches.append(id1)
      else:
          matches.append((id1, min_id2, np.degrees(min_dist)))
  
  time_taken = time.perf_counter() - start
  return matches, no_matches, time_taken


# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # The example in the question
  ra1, dec1 = np.radians([180, 30])
  cat2 = [[180, 32], [55, 10], [302, -44]]
  cat2 = np.radians(cat2)
  ra2s, dec2s = cat2[:,0], cat2[:,1]
  dists = angular_dist(ra1, dec1, ra2s, dec2s)
  print(np.degrees(dists))

  cat1 = np.array([[180, 30], [45, 10], [300, -45]])
  cat2 = np.array([[180, 32], [55, 10], [302, -44]])
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)

  # A function to create a random catalogue of size n
  def create_cat(n):
    ras = np.random.uniform(0, 360, size=(n, 1))
    decs = np.random.uniform(-90, 90, size=(n, 1))
    return np.hstack((ras, decs))

  # Test your function on random inputs
  np.random.seed(0)
  cat1 = create_cat(10)
  cat2 = create_cat(20)
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)
