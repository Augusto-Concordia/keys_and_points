# import other necessary libaries
from utils import create_line, create_mask

# load the input image

# run Canny edge detector to find edge points

# create a mask for ROI by calling create_mask

# extract edge points in ROI by multipling edge map with the mask

# perform Hough transform

# find the right lane by finding the peak in hough space

# zero out the values in accumulator around the neighborhood of the peak

# find the left lane by finding the peak in hough space

# plot the results
