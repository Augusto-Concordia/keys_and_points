import cv2
import numpy as np


def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    # YOUR CODE HERE
    # You can use skimage or OpenCV to perform SIFT matching
    sift: cv2.SIFT = cv2.SIFT_create()
    kps1, desc1 = sift.detectAndCompute((I1 * 255).astype(np.uint8), None)
    kps2, desc2 = sift.detectAndCompute((I2 * 255).astype(np.uint8), None)

    locs1 = np.zeros((len(kps1), 2), dtype=float)
    locs2 = np.zeros((len(kps2), 2), dtype=float)

    # transform the keypoints to a numpy array of locations
    for i, kp in enumerate(kps1):
        locs1[i, :] = kp.pt

    for i, kp in enumerate(kps2):
        locs2[i, :] = kp.pt

    # BF Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    dMatches = bf.match(desc1, desc2)

    matches = np.array([[m.queryIdx, m.trainIdx] for m in dMatches])

    # END YOUR CODE

    return matches, locs1, locs2


def computeH_ransac(matches, locs1, locs2):
    # Compute the best fitting homography using RANSAC given a list of matching pairs

    # YOUR CODE HERE
    # You should implement this function using Numpy only

    SSD_THRESHOLD = 6.0
    ITERATION_LIMIT = 100

    def buildAMatrix(indices) -> np.ndarray:
        matrixA = np.zeros((len(indices) * 2, 9))

        for i, matchIndex in enumerate(indices):
            # get the corresponding 4 random points
            fp, fpp = matches[matchIndex, :]

            # get their positions
            x, y = locs1[fp, :]
            xp, yp = locs2[fpp, :]

            # create the matrix A for homography (p' = Hp)
            matrixA[i * 2, :] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
            matrixA[i * 2 + 1, :] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

        return matrixA

    inliers = np.zeros((len(matches), 1), dtype=int)
    highestInliers = 0
    currentIteration = 0
    bestIteration = 0

    for _ in range(ITERATION_LIMIT):
        random_selection = np.random.choice(len(matches), 4, replace=False)

        # create the matrix A for homography (p' = Hp)
        matrixA = buildAMatrix(random_selection)

        # compute the SVD of P (to get the h for these set of points)
        _, _, V = np.linalg.svd(matrixA)
        h = np.reshape(V[-1, :], (3, 3))

        # compute the transformed points
        currentInliers = np.zeros((len(matches), 1), dtype=int)
        currentInliersCount = 0

        # compute the SSD (sum of squared differences) for each transformed point (from the 1st image) and the corresponding point in the 2nd image
        for i, match in enumerate(matches):
            # get the point from the original image (the one we want to transform)
            x, y = locs1[match[0], :]

            # transform the point using the computed homography
            transformedPoint = h @ [x, y, 1]

            # transformed point without the homogeneous coordinate
            nonHomoPoint = (transformedPoint / transformedPoint[2])[:2]

            # target point
            targetPoint = locs2[match[1], :]

            # compute the SSD for this point and the corresponding point in the target image
            ssd = np.sum((targetPoint - nonHomoPoint) ** 2)

            if ssd >= SSD_THRESHOLD:  # we don't want outliers
                currentInliers[i, :] = -1
                continue

            currentInliers[i, :] = i
            currentInliersCount += 1

        if currentInliersCount > highestInliers:
            highestInliers = currentInliersCount
            inliers = currentInliers[currentInliers != -1]
            bestIteration = currentIteration

        currentIteration += 1

    # we have found the best set of inliers, now we can compute the best H for these points

    # create the matrix A for final homography (p' = Hp)
    matrixA = buildAMatrix(inliers)

    # compute the SVD of A (to get the best H for these set of points)
    _, _, V = np.linalg.svd(matrixA)
    bestH = np.reshape(V[-1, :], (3, 3))

    print(
        f"Found best H after {bestIteration} iterations with {highestInliers} inliers. Homography: {bestH}")

    # END YOUR CODE

    return bestH, inliers


def compositeH(H, template, img):
    # Create a composite image after warping the template image on top
    # of the image using homography

    # Create mask of same size as template
    mask = np.ones(template.shape)

    # Warp mask by appropriate homography
    mask = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))
    inverted_mask = cv2.bitwise_not(mask)

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(
        template, H, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    img = (mask * warped_template + cv2.bitwise_and(inverted_mask, img))

    return img
