# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:27:42 2018

@author: pierr
"""
import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir('C:\\Users\pierr\Desktop\Data Science - Astronomy\Decision Trees')

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    u_g = data['u'] - data['g']
    r_i = data['r'] - data['i']
    
    # Make a redshift array
    redshift = data['redshift']
    
    # Create the plot with plt.scatter and plt.colorbar
    _ = plt.scatter(u_g, r_i, s=0.5, lw=0, c=redshift, cmap=cmap)
    cb = plt.colorbar(_)
    cb.set_label('Redshift')
    
    # Define your axis labels and plot title
    plt.xlabel('Colour index u-g')
    plt.ylabel('Colour index r-i')
    plt.title('Redshift (colour) u-g versus r-i')
    
    # Set any axis limits
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1)
    
    plt.show()