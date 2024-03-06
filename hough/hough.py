# import other necessary libaries
from utils import create_line, create_line_endpoints, create_mask
import cv2
import numpy as np

def show_matlike_and_wait(title, mat):
    cv2.imshow(title, cv2.extractChannel(mat, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough_transform(edges: cv2.typing.MatLike, rho_res: int, theta_res: float) -> cv2.typing.MatLike:
    """
    Applies the Hough transform on the given edges to detect lines.

    Args:
        edges (cv2.typing.MatLike): The input edge map.
        rho_res (int): The resolution of the rho parameter in pixels.
        theta_res (float): The resolution of the theta parameter in degrees.

    Returns:
        cv2.typing.MatLike: The accumulator matrix containing the detected lines.
    """

    # create the accumulator matrix
    diagonal = np.sqrt(edges.shape[0]**2 + edges.shape[1]**2) # largest possible rho value
    accum_h = np.zeros((int(180 / theta_res), int(diagonal * rho_res)), dtype=np.uint8)

    # iterate over each edge point
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            if edges[x, y] == 0:
                continue

            # iterate over each theta value
            for theta in range(0, 180, int(theta_res)):
                # calculate rho value
                rho = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                accum_h[theta, rho] += 1
    
    return accum_h

def main():
    # load the input image
    image = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)
    image_h, image_w = image.shape
 
    # run Canny edge detector to find edge points
    edges = cv2.Canny(image, 5, 150)

    # create a mask for ROI by calling create_mask
    mask = create_mask(image_h, image_w)
    mask = (mask * 255.0).astype(np.uint8) # convert to 8-bit (grayscale) image

    # extract edge points in ROI by multiplying edge map with the mask
    roi_edges = cv2.bitwise_and(edges, mask)

    # perform Hough transform
    rho_res = 2
    theta_res = 1.0
    hough_space = hough_transform(roi_edges, rho_res, theta_res)
    print(f'Hough space shape: {hough_space.shape}')

    # find the right lane by finding the peak in hough space
    peak_idx = np.unravel_index(np.argmax(hough_space, axis=None), hough_space.shape) # find the index of the peak value
    peak = hough_space[peak_idx] # find the peak value
    print(f'Peak value: {peak}, peak index: {peak_idx}')
    
    # get the rho and theta values of the peak
    rho = peak_idx[1] * rho_res
    theta = peak_idx[0] * theta_res
    print(f'Rho: {rho}, Theta: {theta}')

    # display the line on the original image
    line_endpoints = create_line_endpoints(rho, theta, image)
    x_scalar = hough_space.shape[1] / (image.shape[0] * rho_res) # scale factor for x-axis (to account for the resolution of rho parameter)
    scaled_line_endpoints = (int(line_endpoints[0] * x_scalar), int(line_endpoints[1]), int(line_endpoints[2] * x_scalar), int(line_endpoints[3])) # scale the line to the original image size
    image = cv2.line(image, scaled_line_endpoints[0:2], scaled_line_endpoints[2:4], (0, 255, 0), 2)
    show_matlike_and_wait('Image with detected line', image)

    # zero out the values in accumulator around the neighborhood of the peak

    # find the left lane by finding the peak in hough space

    # plot the results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted: bye!')