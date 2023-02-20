from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
import cv2
import numpy as np


from py_floor_plan_segmenter.segment import make_binary, crop_single_info, crop_background, make_invert, remove_small_connected_components
from py_floor_plan_segmenter.debugging.debugging_factory import debugger


def find_map_orientation_adjustment(im):
    # im: assumed to be a grayscale image
    contours, _ = cv2.findContours(np.uint8(im*255.), 1, 2)
    # Get the second largest bounding box, which is always the map.
    rect = cv2.minAreaRect(contours[-2])
    angle = rect[2]
    return rect, angle


def rotate_image(image, angle):
    """This function rotates an image in a non-pixel-perfect way.

    The background of the result will be equal to the first pixel of the image.

    Args:
        image (numpy.ndarray): The input image. The dtype should be either np.float32 or np.uint8.
        angle (float): The amount of angle degrees to rotate.

    Returns:
        np.ndarray: A numpy array of dtype np.float32 including the rotated image. There is no pixel perfection.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    if (image.dtype == np.float32):
        image_uint8 = np.uint8(image*255.)
    else:
        image_uint8 = image

    result = cv2.warpAffine(
        image_uint8, rot_mat, image_uint8.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=int(image_uint8[0, 0]))

    return np.float32(result/255.)


def draw_bb(im, rect, color=(0, 0, 1.)):
    output = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(output, [box], 0, color, 2)
    return output


def draw_line(im, x0, y0, x1, y1, color=(1., 0, 0), line_thickness=1):
    cv2.line(im, (x0, y0), (x1, y1), color, thickness=line_thickness)
    return im


def compute_weighted_angle_histogram(rank, debug=False, **config):
    edges = canny(rank, **config["canny"])
    debugger.add("unrotated_edges", edges)

    lines = probabilistic_hough_line(
        edges, **config["probabilistic_hough_line"])

    if debug:
        lines_fig = cv2.cvtColor(np.zeros_like(rank), cv2.COLOR_GRAY2BGR)

    hist = np.zeros((config["resolution"],))
    for line in lines:
        p0, p1 = line

        x0, y0 = p0
        x1, y1 = p1

        angle = np.arctan2(y1-y0, x1-x0) / np.pi * 180
        angle_90 = angle % 90

        if debug:
            lines_fig = draw_line(lines_fig, x0, y0, x1, y1,
                                  color=np.array((1, 0, 0))*angle_90/90)

        angle_bin = int(np.floor(angle_90 / 90 * config["resolution"]))
        length = np.sqrt(np.power(x1-x0, 2) + np.power(y1-y0, 2))
        hist[angle_bin] += length
    if debug:
        debugger.add("unrotated_lines", lines_fig)
    return hist


def find_best_alignment_angle_from_histogram(hist, **config):
    k = config["window_half_size"]
    # Full window size is:
    N = 2*k+1
    augmented_hist = np.hstack([hist[-k:], hist, hist[:k]])
    sliding_average = np.mean(np.lib.stride_tricks.sliding_window_view(
        augmented_hist, window_shape=N), axis=1)

    debugger.add("angle_histogram_sliding_average", sliding_average)

    theta_deg = np.argmax(sliding_average)/config["resolution"] * 90
    if theta_deg > 45:
        theta_deg -= 90
    return theta_deg


def find_alignment_angle(grayscale, **config):
    # Create a local binary just to figure out the crop_info
    binary_for_crop = make_binary(
        grayscale, **config["binary_for_crop/make_binary"])
    crop_info = crop_single_info(binary_for_crop, **config["crop_single_info"])

    # cropped: Crop rank map to ROI
    rank = crop_background(grayscale, crop_info)
    debugger.add("unrotated", rank)

    # binary: Make image into binary, to remove some noise
    rank = make_binary(rank, **config["binary/make_binary"])

    # invert1: invert the image
    rank = make_invert(rank)
    # components_out: Remove small connected components from the outside
    rank = remove_small_connected_components(
        rank, **config["components_out/remove_small_connected_components"])
    # invert2: invert the image
    rank = make_invert(rank)
    # components_in: Remove small connected components from the inside
    rank = remove_small_connected_components(
        rank, **config["components_in/remove_small_connected_components"])

    # rank: Make image into binary
    rank = make_binary(rank, **config["rank/make_binary"])
    debugger.add("unrotated_processed", rank)

    angle_histogram = compute_weighted_angle_histogram(
        rank, debug=debugger.debug, **config["compute_weighted_angle_histogram"])
    debugger.add("angle_histogram", angle_histogram)

    angle_deg = find_best_alignment_angle_from_histogram(
        angle_histogram, **config["find_best_alignment_angle_from_histogram"])

    return angle_deg
