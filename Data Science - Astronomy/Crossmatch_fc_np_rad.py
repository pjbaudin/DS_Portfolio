# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 09:17:21 2018

@author: pierr
"""
import numpy as np
from time import time

def angular_dist_rad(r1, d1, r2, d2):
    deltar = np.abs(r1 - r2)
    deltad = np.abs(d1 - d2)
    angle = 2*np.arcsin(np.sqrt(np.sin(deltad/2)**2 
                        + np.cos(d1)*np.cos(d2)*np.sin(deltar/2)**2))
    return angle

def crossmatch(coords1, coords2, max_radius):
    start_time = time()
    max_radius = np.radians(max_radius)
    matches = []
    no_matches = []
    
    # Convert coordinates to radians
    coords1 = np.radians(coords1)
    coords2 = np.radians(coords2)
    # Find ascending declination order of second catalogue
    asc_dec = np.argsort(coords2[:, 1])
    coords2_sorted = coords2[asc_dec]
    dec2_sorted = coords2_sorted[:, 1]
    
    for id1, (ra1, dec1) in enumerate(coords1):
        closest_dist = np.inf
        closest_id2 = None
        
        # Declination search box
        min_dec = dec1 - max_radius
        max_dec = dec1 + max_radius
        
        # Start and end indices of the search
        start = dec2_sorted.searchsorted(min_dec, side='left')
        end = dec2_sorted.searchsorted(max_dec, side='right')
        
        for s_id2, (ra2, dec2) in enumerate(coords2_sorted[start:end+1], start):
            dist = angular_dist_rad(ra1, dec1, ra2, dec2)
            if dist < closest_dist:
                closest_sorted_id2 = s_id2
                closest_dist = dist
        
        # Ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            no_matches.append(id1)
        else:
            closest_id2 = asc_dec[closest_sorted_id2]
            matches.append([id1, closest_id2, np.degrees(closest_dist)])
    
    time_taken = time() - start_time
    return matches, no_matches, time_taken

# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # The example in the question
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