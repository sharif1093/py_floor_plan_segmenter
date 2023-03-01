import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.ndimage import label
import skimage
# from skimage.filters import meijering, sato, frangi, hessian
from skimage.filters import meijering
from skimage.segmentation import watershed


def load_map_from_file_uint8(path: Path):
    """This function loads a map file.

    Args:
        path (Path): The path to the map.

    Returns:
        np.ndarray: The 3-channel RGB loaded map with dtype (np.uint8) and size (width, height, 3)
    """
    return np.asarray(cv2.imread(str(path)), dtype=np.uint8)


def load_map_from_file(path: Path):
    """This function loads a map file.

    Args:
        path (Path): The path to the map.

    Returns:
        np.ndarray: The 3-channel RGB loaded map with dtype (np.float32) and size (width, height, 3)
    """
    return np.asarray(cv2.imread(str(path)), dtype=np.float32) * 1 / 255.


def load_map_from_buffer(buffer):
    nparr = np.frombuffer(buffer, np.uint8)
    return np.asarray(cv2.imdecode(nparr, cv2.IMREAD_COLOR), dtype=np.float32) * 1 / 255.


def make_gray(src: np.ndarray):
    """Converts a 3-channel RGB map to a 1-channel grayscale map.

    Args:
        src (np.ndarray): The input image in dtype (np.float32) and size (width, height, 3)

    Returns:
        np.ndarray: The output map in dtype (np.float32) and size (width, height)
    """
    return cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


def load_gray_map_from_file(path: Path):
    # load_map_from_file: Load the image from name
    # make_gray: Convert to 1-channel grayscale
    return make_gray(load_map_from_file(path))


def load_gray_map_from_buffer(buffer):
    return make_gray(load_map_from_buffer(buffer))


def find_bb(map: np.ndarray):
    map_uint8 = np.asarray(map * 255, dtype=np.uint8)
    width = map.shape[1]
    height = map.shape[0]

    background = map_uint8[0, 0]
    padding = 1

    # Find x1
    x1 = 0
    for x in range(width):
        if np.all(map_uint8[:, x] == background):
            continue
        x1 = max(x-padding, 0)
        break

    # Find x2
    x2 = width
    for x in range(width-1, -1, -1):
        if np.all(map_uint8[:, x] == background):
            continue
        x2 = min(x+padding, width)
        break

    # Find y1
    y1 = 0
    for y in range(height):
        if np.all(map_uint8[y, :] == background):
            continue
        y1 = max(y-padding, 0)
        break

    # Find y2
    y2 = height
    for y in range(height-1, -1, -1):
        if np.all(map_uint8[y, :] == background):
            continue
        y2 = min(y+padding, height)
        break

    rectangle = (x1, y1, x2, y2)
    return rectangle


def crop_single_info(map: np.ndarray, padding: int = 0):
    shape = map.shape
    width = shape[1]
    height = shape[0]

    rect = find_bb(map)

    x1, y1, x2, y2 = rect

    a = max(x1-padding, 0)
    b = max(y1-padding, 0)
    c = min(width - x2 - padding, width)
    d = min(height - y2 - padding, height)
    rect = (a, b, c, d)
    return rect


def crop_bundle_info(rank_map: np.ndarray, track_map: np.ndarray):
    rank_shape = rank_map.shape
    track_shape = track_map.shape

    if (rank_shape != track_shape):
        raise ValueError(
            f"The two maps must be same size. {rank_shape}!={track_shape}")

    width = rank_shape[1]
    height = rank_shape[0]

    rect1 = find_bb(rank_map)
    rect2 = find_bb(track_map)

    x1, y1, x2, y2 = rect1
    a1, b1, a2, b2 = rect2
    rect = (min(x1, a1), min(y1, b1), width -
            max(x2, a2), height - max(y2, b2))
    return rect


def crop_background(map: np.ndarray, rectangle):
    x1, y1, x2, y2 = rectangle
    return map[y1:-y2, x1:-x2]


def uncrop_background(map: np.ndarray, rectangle, background):
    width = map.shape[1]
    height = map.shape[0]
    x1, y1, x2, y2 = rectangle

    shape = (y1+height+y2, x1+width+x2)
    uncropped = np.full(shape, fill_value=background, dtype=map.dtype)
    uncropped[y1:-y2, x1:-x2] = map

    return uncropped


def remap_border(map: np.ndarray, border_label: int = None):
    if (border_label == None):
        border_label = np.max(map)
    res = deepcopy(map)
    res[map == border_label] = 255
    return res


def make_binary(src: np.ndarray, threshold: float):
    """Converts a map into binary

    Args:
        src (np.ndarray): The input map in dtype (np.float32) and size (width, height)
        threshold (float): The value to be used as the binary threshold.

    Returns:
        np.ndarray: The output binary image in dtype (np.float32) and size (width, height)
    """
    return cv2.threshold(src, threshold, 1.0, cv2.THRESH_BINARY)[1]


def dilate_binary(src: np.ndarray, kernel_size: int = 3, iterations: int = 1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(src, kernel, iterations=iterations)


def remove_small_connected_components(src: np.ndarray, method: str = "fixed", min_size: int = 1, connectivity: int = 4, debug: bool = False):
    """A function to remove small connected components below a threshold in area.

    Args:
        src (np.ndarray): The input map in dtype (np.float32) and size (width, height)
        method (str, optional): The method to determine the threashold. Can be "fixed", "mean", or "median".
        min_size (int, optional): Minimum acceptable size of a connected component. Defaults to 1.
        connectivity (int, optional): The connectivity of the connected components which can be 4 or 8. Defaults to 4.
        debug (bool, optional): Whether to output the debug image or not. Defaults to False.

    Returns:
        (np.ndarray, np.ndarray): The tuple of result image and the visualization.
    """

    # Verifications
    if method not in ["fixed", "mean", "median"]:
        raise ValueError("method not equal to fixed, mean, nor median")
    if connectivity not in [4, 8]:
        raise ValueError("connectivity is not 4 nor 8")

    # Find all connected "white" blobs in the binary map.
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(src*255), connectivity, cv2.CV_32S)

    # Remove background first
    nb_blobs -= 1
    stats: np.ndarray = stats[1:]
    centroids = centroids[1:]

    # Find the threshold based on the method used.
    sizes = stats[:, cv2.CC_STAT_AREA]
    if (method == "fixed"):
        threshold = min_size
    elif (method == "mean"):
        threshold = np.mean(sizes)
    elif (method == "median"):
        threshold = np.median(sizes)

    # Keep only blobs with large enough area.
    # Using "visual" for visualizations.
    result = np.zeros((src.shape), dtype=np.float32)
    if debug:
        visual = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for blob in range(nb_blobs):
        if debug:
            x = stats[blob, cv2.CC_STAT_LEFT]
            y = stats[blob, cv2.CC_STAT_TOP]
            w = stats[blob, cv2.CC_STAT_WIDTH]
            h = stats[blob, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 1.0, 0), 3)
            # Visualize centroid of blob
            # (cX, cY) = centroids[blob]
            # cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        # Keep blob if that is above threshold
        if sizes[blob] >= threshold:
            # Use "blob + 1" because we have removed the background.
            result[im_with_separated_blobs == blob + 1] = 1.0

    if debug:
        return result, visual
    else:
        return result


def generate_ridge_map_meijering(src: np.ndarray, mode: str = 'constant', sigmas=[2]):
    """This is a ridge generation function which uses skimage and resemples Mathematica RidgeDetect more.

    Args:
        src (np.ndarray): The input map in dtype (np.float32) and size (width, height).
        mode (str, optional): The boundary condition. Can be {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}. Defaults to 'nearest'.
        sigmas (list, optional): _description_. Defaults to [2].

    Returns:
        np.ndarray: The output map in dtype (np.float32) and size (width, height).
    """
    return meijering(src, sigmas=sigmas, mode=mode)


def generate_gaussian_seeds_skimage(src: np.ndarray, background_mask, sigma: float, threshold: float = 0.05):
    """A function to generate Gaussian seeds for a binary map

    Args:
        src (np.ndarray): The input map in dtype (np.float32) and size (width, height)
        sigma (float, optional):
        threshold (float): The threshold used to binarize the image. Default to 0.05.
        background_erosion_kernel_size (int): The kernel size used for eroding the background. The greater the number the more
            certain about about being part of the background. Default to 5;


    Returns:
        np.ndarray: The output map in dtype (np.int32) and size (width, height). It is a label image. Background is labeld as 255.
    """
    result = skimage.filters.gaussian(
        src, sigma=sigma, mode='nearest', truncate=4)
    res_bin = make_binary(result, threshold)

    # Find labels (numbering each connected segment separately)
    res_bin_uint8 = 255 - np.uint8(res_bin*255)
    lbl, _ = label(res_bin_uint8)

    # Change background label to 255
    lbl[background_mask] = 255

    return lbl


def create_watershed_segment_skimage(src: np.ndarray, labels: np.ndarray, connectivity: int = 4):
    """Creates the watershed segmentation based on provided labels. It uses skimage's version of watershed

    Args:
        src (np.ndarray): The input map in dtype (np.float32) and size (width, height) or size (width, height, 3)
        labels (np.ndarray): The labels for the input seeds of the watershed algorithm. Label 255 is assumed for the background.

    Returns:
        np.ndarray: The segmentation of the input image in dtype (np.uint8) and size (width, height). Background is labeled as 0.
            The label for the edges is equal to maximum value of the label image.
    """
    labels_local = deepcopy(labels)
    segments = watershed(
        src, labels_local, connectivity=connectivity, watershed_line=True)

    # segments == 0 ---> Watershed lines
    # Change border labels to the number of unique elements + 1, which will become the new max then.
    segments[segments == 0] = len(np.unique(segments)) - 1
    # Change the background to 0
    segments[segments == 255] = 0

    return segments.astype(np.uint8)


def make_invert(src: np.ndarray):
    if (src.dtype == np.uint8):
        return 255 - src
    elif (src.dtype == np.float32):
        return 1. - src
    else:
        raise TypeError(
            "make_invert does not support the provided type: {}".format(src.dtype))
