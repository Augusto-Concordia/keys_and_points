def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching
    
    ### END YOUR CODE
    
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    # Compute the best fitting homography using RANSAC given a list of matching pairs
    
    ### YOUR CODE HERE
    ### You should implement this function using Numpy only
    
    ### END YOUR CODE

    return bestH, inliers

def compositeH(H, template, img):

    # Create a compositie image after warping the template image on top
    # of the image using homography


    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image
    
    return composite_img
