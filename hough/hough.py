# import other necessary libaries
import typing
from matplotlib import pyplot as plt
from utils import create_line, create_mask
import cv2
import numpy as np


def show_matlike_and_wait(title, mat):
    cv2.imshow(title, cv2.extractChannel(mat, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hough_transform(edges: cv2.typing.MatLike, theta_step: int, rho_step: int) -> tuple[cv2.typing.MatLike, np.ndarray, np.ndarray]:
    edgesY, edgesX = edges.shape

    # create the accumulator matrix
    # largest possible rho value
    diagonal = int(np.sqrt(edgesX ** 2 + edgesY ** 2))
    thetas = np.deg2rad(np.arange(0, 180, theta_step))  # range of theta values
    rhos = np.arange(0, diagonal, rho_step)  # range of rho values

    accum_h = np.zeros((len(thetas), len(rhos)), dtype=np.float16)

    # iterate over each edge point
    for y in range(edgesY):
        for x in range(edgesX):
            if edges[y, x] == 0:
                continue

            # iterate over each theta value
            for t_idx in range(len(thetas)):
                # calculate rho value
                rho = int(
                    np.round(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx])))
                accum_h[t_idx, rho] += 1

    return accum_h, thetas, rhos


def max_2d_idx(mat: cv2.typing.MatLike) -> tuple[tuple[np.intp, ...], typing.Any]:
    peak_idx = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
    peak = mat[peak_idx]  # find the peak value

    return peak_idx, peak


def main():
    # prep plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))

    # load the input image
    image = cv2.imread('road.jpg', cv2.IMREAD_ANYCOLOR)
    image_h, image_w, _ = image.shape

    # run Canny edge detector to find edge points
    edges = cv2.Canny(image, 5, 150)
    ax1.imshow(edges, cmap='gray')
    ax1.set_title('Canny edge map')
    ax1.axis('off')

    # create a mask for ROI by calling create_mask
    mask = create_mask(image_h, image_w)
    # convert mask to 8-bit (grayscale) image
    mask = (mask * 255.0).astype(np.uint8)
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')
    ax2.axis('off')

    # extract edge points in ROI by multiplying edge map with the mask
    roi_edges = cv2.bitwise_and(edges, mask)
    ax3.imshow(roi_edges, cmap='gray')
    ax3.set_title('ROI edges')
    ax3.axis('off')

    # perform Hough transform
    rho_step = 1
    theta_step = 1
    hough_space, thetas, rhos = hough_transform(
        roi_edges, theta_step, rho_step)
    print(f'Hough space shape: {hough_space.shape}')

    print('Right lane properties:')
    # find the right lane by finding the peak in hough space
    # find the index of the peak value
    peak_idx, peak = max_2d_idx(hough_space)
    print(f'\tPeak value: {peak}, Peak index: {peak_idx}')

    # get the rho and theta values of the peak
    theta = thetas[peak_idx[0]]
    rho = rhos[peak_idx[1]]
    print(f'\tRho: {rho}, Theta: {theta}')

    # display the line on the original image
    x1 = int(image_w / 2)
    y1 = int(rho / np.sin(theta) - np.cos(theta) / np.sin(theta) * x1)
    x2 = image_w
    y2 = int(rho / np.sin(theta) - np.cos(theta) / np.sin(theta) * x2)
    print(f'\tLine endpoints: {x1, y1} and {x2, y2}')

    # update image with the detected line
    image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # BGR

    # zero out the values in accumulator around the neighborhood of the peak
    clear_radius = 50
    for i in range(peak_idx[0] - clear_radius, peak_idx[0] + clear_radius):
        for j in range(peak_idx[1] - clear_radius, peak_idx[1] + clear_radius):
            hough_space[i, j] = 0

    # find the left lane by finding the peak in hough space
    print('Left lane properties:')
    peak_idx, peak = max_2d_idx(hough_space)
    print(f'\tPeak value: {peak}, Peak index: {peak_idx}')

    # get the rho and theta values of the peak
    theta = thetas[peak_idx[0]]
    rho = rhos[peak_idx[1]]
    print(f'\tRho: {rho}, Theta: {theta}')

    # display the line on the original image
    x1 = 0
    y1 = int(rho / np.sin(theta) - np.cos(theta) / np.sin(theta) * x1)
    x2 = int(image_w / 2)
    y2 = int(rho / np.sin(theta) - np.cos(theta) / np.sin(theta) * x2)
    print(f'\tLine endpoints: {x1, y1} and {x2, y2}')

    image = cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # BGR
    ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax4.set_title('Detected lane')
    ax4.axis('off')

    # plot the results
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted: bye!')
