from skimage import color
import cv2
import copy
import numpy as np
import random
from pathlib import Path
import networkx as nx


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


try:
    import matplotlib
    from matplotlib import pyplot as plt
    if not is_notebook():
        matplotlib.use('Agg')
    else:
        print("Running in an interactive notebook!")
except ModuleNotFoundError:
    # Error handling
    pass


#############################
# Image operation functions #
#############################


def show_image(ax, img, gray=True, title=None, fontsize=None):
    if gray:
        ax.imshow(img, cmap=plt.cm.gray)
    else:
        ax.imshow(img)

    if title is not None:
        if fontsize is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=fontsize)


def highlight(src, target, color, alpha):
    # Target: A binary image
    src_rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

    # If grayscale image, just convert to rgb
    if (len(target.shape) == 2):
        target_rgb = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    else:
        target_rgb = target.copy()

    [r, g, b] = cv2.split(target_rgb)
    target_colored = cv2.merge((b*color[0],
                                g*color[1],
                                r*color[2]))

    result = cv2.addWeighted(src_rgb, 1-alpha, target_colored, alpha, 0, None)
    return result


def convert_uint8_to_rgb_random(target, exclude=[]):
    # target: Must be a grayscale image with type np.uint8 (CV_8UC1)
    #         Each color in target will be converted to a unique color.
    labels = np.unique(target)
    label_map = {i: labels[i] for i in range(len(labels))}
    colors = [(i/(len(labels)+2) * 360, 200, 200) for i in range(len(labels))]
    random.shuffle(colors)

    target_rgb = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)

    for i in range(len(labels)):
        if not label_map[i] in exclude:
            target_hsv[target == label_map[i]] = colors[i]
        else:
            target_hsv[target == label_map[i]] = (120, 120, 0)

    target_rgb = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2RGB)
    return np.float32(target_rgb * 1./255)


def convert_uint8_to_rgb_random_rgb(target, exclude=[]):
    # target: Must be a grayscale image with type np.uint8 (CV_8UC1)
    #         Each color in target will be converted to a unique color.
    labels = np.unique(target)
    label_map = {i: labels[i] for i in range(len(labels))}
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(1, labels.max() + 1)]

    target_rgb = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    # target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)

    for i in range(len(labels)):
        if label_map[i] in exclude:
            target_rgb[target == label_map[i]] = (0, 0, 0)
        # elif label_map[i] in black:
        #     target_rgb[target == label_map[i]] = (0, 0, 0)
        else:
            target_rgb[target == label_map[i]] = colors[i]
    return np.float32(target_rgb * 1./255)


def convert_uint8_to_rgb_colormap(src: np.ndarray, exclude_zero=True, colormap=cv2.COLORMAP_JET, remap={}):
    """Converts a label image to color image based on a deterministic color map.

    Args:
        src (np.ndarray): The source image in dtype (np.float32|np.uint8) and size (width, height)
        exclude_zero (bool, optional): Whether to exclude label 0 or not. Defaults to True.
        colormap (_type_, optional): The colormap to use. Can be vaule such as COLORMAP_JET | COLORMAP_HSV | COLORMAP_RAINBOW.
            Defaults to cv2.COLORMAP_JET.
        remap (bool, optional): Whether to remap unique values in the src to be near each other. Needs src to be np.uint8.

    Returns:
        np.ndarray: The segmentation of the input image in dtype (np.float32) and size (width, height, 3).
    """
    # Remap just in case!
    src_mapped = np.zeros_like(src, dtype=src.dtype)
    unique = np.unique(src)
    for u in unique:
        if u in remap:
            src_mapped[src == u] = remap[u]
        else:
            src_mapped[src == u] = u

    if np.max(src_mapped) > 0:
        src_normalized = (src_mapped - np.min(src_mapped)) * \
            1./np.max(src_mapped)
    else:
        src_normalized = np.zeros_like(src_mapped)
    src_normalized_uint8 = np.uint8(src_normalized * 255)

    src_colored = cv2.applyColorMap(src_normalized_uint8, colormap)

    if exclude_zero:
        # Set background cells' colors to black again.
        src_colored[src == 0] = (0, 0, 0)

    return np.float32(src_colored * 1./255)


def gen_highlighted_segments(segments, cropped):
    # segments_color = convert_uint8_to_rgb_random(segments, [0])
    segments_color = convert_uint8_to_rgb_colormap(segments)
    highlighted_segments = highlight(
        cropped, segments_color, (1., 1., 1.), 0.6)
    return highlighted_segments


def gen_highlighted_seeds(seeds, dilated):
    remap = {255: len(np.unique(seeds))-1}
    seeds_color = convert_uint8_to_rgb_colormap(np.uint8(seeds), remap=remap)
    highlighted_seeds = highlight(dilated, seeds_color, (1., 1., 1.), 0.6)
    return highlighted_seeds


def visualize_segments_double(output_path: Path, seeds, segments, cropped, dilated, title, name: str):
    highlighted_segments = gen_highlighted_segments(segments, cropped)
    highlighted_seeds = gen_highlighted_seeds(seeds, dilated)

    fig, ax = plt.subplots(1, 2)
    # fig.suptitle(title, fontsize=14)
    show_image(ax[0], highlighted_seeds, title="Superposed seed map")
    show_image(ax[1], highlighted_segments, title="Over-segmented map")
    plt.savefig(output_path / f"{name}.png", dpi=600)


def close_all_figures():
    plt.close('all')


def dump_color_image(src, title, path):
    _, axs = plt.subplots(1, 1)
    show_image(axs, src, title=title)
    # axs.set_title("NCC vs Sigma")
    plt.savefig(path, dpi=150)
    plt.clf()


def plot_segment_debug(output_path, title,
                       unrotated, unrotated_processed, unrotated_edges, unrotated_lines,
                       cropped, binary, rank, ridges,
                       seeds, segments, binary_dilated):

    highlighted_seeds = gen_highlighted_seeds(seeds, binary_dilated)
    highlighted_segments = gen_highlighted_segments(segments, cropped)

    fig, axs = plt.subplots(2, 4, figsize=(12, 12), constrained_layout=True)

    # fig.suptitle(title, fontsize=14)

    fontsize = 18
    show_image(axs[0, 0], unrotated, title="Raw map", fontsize=fontsize)
    show_image(axs[0, 1], unrotated_processed,
               title="Preprocessed", fontsize=fontsize)
    show_image(axs[0, 2], unrotated_edges, title="Edges", fontsize=fontsize)
    show_image(axs[0, 3], unrotated_lines, title="Lines", fontsize=fontsize)

    show_image(axs[1, 0], cropped,  title="Aligned map", fontsize=fontsize)
    show_image(axs[1, 1], binary,  title="Binary map", fontsize=fontsize)
    show_image(axs[1, 2], rank,  title="Denoised binary map",
               fontsize=fontsize)
    show_image(axs[1, 3], ridges,  title="Ridges", fontsize=fontsize)

    # show_image(axs[2, 0], highlighted_seeds,  title="Highlighted Labels")
    # show_image(axs[2, 1], highlighted_segments,
    #            title="Highlighted Segments")
    # show_image(axs[2, 2], binary_dilated,  title="Binary Dilated")

    plt.savefig(output_path / "debug.png", dpi=300)


def plot_sigmas_vs_ncc(sigmas, nccs, path):
    _, axs = plt.subplots(1, 1)
    axs.plot(np.array(sigmas), np.array(nccs))
    axs.set_title("NCC vs Sigma")

    axs.set_xlabel("Sigma")
    axs.set_ylabel("NCC")

    axs.set_xlim([np.min(sigmas), np.max(sigmas)])
    axs.set_ylim([0, np.max(nccs)])

    plt.savefig(path / "ncc_vs_sigma.png", dpi=150)


def plot_hist_and_sliding_average(hist, sliding_average_hist, x, path, title="Histogram of angles"):
    _, axs = plt.subplots(1, 1)

    axs.plot(x, hist)
    axs.plot(x, sliding_average_hist)

    axs.set_title(title)
    axs.set_xlabel("Angle (deg)")
    axs.set_ylabel("Frequency")

    axs.set_ylim([0, max(np.max(hist), np.max(sliding_average_hist))])
    axs.set_xlim([0, np.max(x)])

    plt.savefig(path / "hist_vs_sliding_average.png", dpi=150)


def draw_graph(ax, G: nx.Graph, node_color='red'):
    # Avoid drawing zero
    nx.draw(G,
            ax=ax,
            alpha=0.9,
            with_labels=True,
            font_weight='bold',
            font_color='whitesmoke',
            node_color=node_color,
            edge_color="tab:blue",
            node_size=10 *
            np.array(
                [area for node, area in nx.get_node_attributes(G, "area").items()]),
            width=2,
            font_family='sans-serif',
            pos=nx.get_node_attributes(G, "position"))


def visualize_graph(output_path: Path, G, segments, name):
    _, axs = plt.subplots(1, 1)
    show_image(axs, segments)
    draw_graph(axs, G)
    plt.savefig(output_path/f"{name}.png", dpi=600)


def visualize_common_borders_map(output_path: Path, common_borders_map):
    _, axs = plt.subplots(1, 1)
    show_image(axs, common_borders_map)
    plt.savefig(output_path / "common_borders_map.png", dpi=600)


def labels_2_colored(labels, background=0):
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(1, labels.max() + 1)]
    colored = color.label2rgb(
        labels,  bg_label=background, bg_color=(0, 0, 0), colors=colors)
    bgr = cv2.cvtColor(colored.astype(np.float32), cv2.COLOR_RGB2BGR)
    return bgr


def labels_2_colored_with_background(labels, image, background=0):
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(1, labels.max() + 1)]
    colored = color.label2rgb(
        labels, image=image, image_alpha=1, bg_label=background, bg_color=(0, 0, 0), colors=colors, alpha=0.6, kind='overlay')
    bgr = cv2.cvtColor(colored.astype(np.float32), cv2.COLOR_RGB2BGR)
    return bgr


def dump_results(output_path: Path, segments, cropped, name="final"):
    # Save a pure version
    segments_no_border = copy.deepcopy(segments)
    segments_no_border[segments == np.max(segments)] = 0

    #####
    segments_colored = labels_2_colored(segments_no_border, background=0)
    cv2.imwrite(str(output_path / f"{name}_segment.png"), segments_colored)

    highlighted_segment = labels_2_colored_with_background(
        segments_no_border, 255*cropped, background=0)

    cv2.imwrite(str(output_path / f"{name}_overlay.png"), highlighted_segment)


def visualize_segmentation(output_path: Path, G, segments, cropped, name):
    segments_color = convert_uint8_to_rgb_random_rgb(
        segments, exclude=[0, np.max(np.unique(segments))])
    highlighted_segment = highlight(cropped, segments_color, (1., 1., 1.), 0.6)

    _, ax = plt.subplots(1, 1)
    show_image(ax, highlighted_segment)

    plt.axis('on')
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False,
                    top=True, labeltop=True)
    plt.savefig(output_path / f"{name}_no_graph.png", dpi=600)
    draw_graph(ax, G)
    plt.savefig(output_path / f"{name}.png", dpi=600)
