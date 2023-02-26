from py_floor_plan_segmenter.segment import crop_single_info, crop_background
from py_floor_plan_segmenter.segment import load_map_from_file_uint8, make_gray
import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage import measure
from sklearn import metrics
from skimage import color
from skimage import segmentation


def labels_2_colored(labels, background=0):
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(1, labels.max() + 1)]
    colored = color.label2rgb(
        labels,  bg_label=background, bg_color=(0, 0, 0), colors=colors)
    bgr = cv2.cvtColor(colored.astype(np.float32), cv2.COLOR_RGB2BGR)
    return bgr


def relabel(labels, background=0):
    unique_labels = list(set(np.unique(labels)) - {background})
    shuffle_labels = np.random.permutation(unique_labels)
    label_mapping = {background: background}
    for index in range(len(shuffle_labels)):
        curr_label = unique_labels[index]
        shuf_label = shuffle_labels[index]
        label_mapping[curr_label] = shuf_label

    shuffled = np.vectorize(label_mapping.get)(labels)

    return shuffled


def calculate_all_scores(y_true, y_pred):
    # Contingency Matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    max_in_cols = np.max(contingency_matrix, axis=0)
    max_in_rows = np.max(contingency_matrix, axis=1)

    sum_of_cols = np.sum(contingency_matrix, axis=0)
    sum_of_rows = np.sum(contingency_matrix, axis=1)
    # np.savetxt("contingency.txt", contingency_matrix, fmt="%6d'")

    EPS = 1e-6
    over_segmentation_scores = max_in_rows / (sum_of_rows + EPS)
    over_mixing_scores = max_in_cols / (sum_of_cols + EPS)

    # Metrics: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html
    homogeneity_score = metrics.homogeneity_score(y_true, y_pred)
    completeness_score = metrics.completeness_score(y_true, y_pred)
    v_score = 2 * (homogeneity_score*completeness_score) / \
        (homogeneity_score+completeness_score)

    nmi_score = metrics.normalized_mutual_info_score(y_true, y_pred)
    ami_score = metrics.adjusted_mutual_info_score(y_true, y_pred)

    fmi_score = metrics.fowlkes_mallows_score(y_true, y_pred)
    ari_score = metrics.adjusted_rand_score(y_true, y_pred)

    return np.mean(over_segmentation_scores), completeness_score, np.mean(over_mixing_scores), homogeneity_score, v_score, nmi_score, ami_score, ari_score, fmi_score


def print_metric(name, value, sanity):
    print(f"{name}: {value:1.5f} (sanity={sanity:1.2f})")


if __name__ == "__main__":
    ############################################################
    # CLI PARSER #
    ##############
    parser = argparse.ArgumentParser(
        description="CLI interface for the 'evaluate' submodule")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="The path to the input directory.")
    parser.add_argument("-g", "--ground-truth", type=Path, required=True,
                        help="The path to the ground truth label.")
    args = parser.parse_args()

    #############################
    ## Verifying CLI arguments ##
    #############################
    # Input file should exist
    if not args.input.is_dir():
        print(f"Input folder {args.input} does not exist!")
        exit(1)
    else:
        input_path = args.input
        base_name = args.input.name

        rank_file = args.input / "global_prior_map.pgm"
        if not rank_file.is_file():
            print(f"global_prior_map.pgm does no exist in the {args.input}!")
            exit(1)

        ground_truth_file = args.ground_truth / "label.png"
        if not ground_truth_file.is_file():
            print(f"label.png does no exist in the {args.input}!")
            exit(1)

    print(f"  .. input: {input_path}")

    rank_labels = make_gray(load_map_from_file_uint8(rank_file))
    ground_truth_mask = make_gray(load_map_from_file_uint8(ground_truth_file))

    # Crop
    crop_info = crop_single_info(ground_truth_mask, padding=10)
    ground_truth_mask = crop_background(ground_truth_mask, crop_info)
    rank_labels = crop_background(rank_labels, crop_info)

    # Remove the boundaries
    rank_labels[rank_labels == 255] = 0

    # rank_labels = measure.label(rank_labels, background=0)
    ground_truth_labels = measure.label(ground_truth_mask, background=0)

    cv2.imwrite("z_rank.png", labels_2_colored(rank_labels, background=0))
    cv2.imwrite("z_gt.png", labels_2_colored(
        ground_truth_labels, background=0))

    rank_labels, _, _ = segmentation.relabel_sequential(rank_labels)


    cv2.imwrite("z_rank_remapped.png", labels_2_colored(
        rank_labels, background=0))

    ground_truth_labels_ravel = ground_truth_labels.ravel()
    rank_labels_ravel = rank_labels.ravel()

    # Remove background cells that are in the GT
    gt_background_cells = ground_truth_labels_ravel == 0
    gt_foreground_cells = ground_truth_labels_ravel != 0
    ground_truth_labels_ravel_foreground = ground_truth_labels_ravel[gt_foreground_cells]
    rank_labels_ravel_foreground = rank_labels_ravel[gt_foreground_cells]

    # Calculate over painting
    rank_labels_ravel_background = rank_labels_ravel[gt_background_cells]
    non_zero_rank_labels_ravel_background = rank_labels_ravel_background[
        rank_labels_ravel_background != 0]
    over_painting = len(non_zero_rank_labels_ravel_background) / \
        len(ground_truth_labels_ravel_foreground)
    print(f"over_painting: {over_painting:1.5f}")

    # Remove background cells that are in the Rank
    rk_background_cells = rank_labels_ravel_foreground == 0
    rk_foreground_cells = rank_labels_ravel_foreground != 0
    rank_labels_ravel_foreground_foreground = rank_labels_ravel_foreground[
        rk_foreground_cells]
    ground_truth_labels_ravel_foreground_foreground = ground_truth_labels_ravel_foreground[
        rk_foreground_cells]

    # Calculate under painting
    rank_labels_ravel_foreground_background = rank_labels_ravel_foreground[
        rk_background_cells]
    under_painting = len(rank_labels_ravel_foreground_background) / \
        len(ground_truth_labels_ravel_foreground)
    print(f"under_painting: {under_painting:1.5f}")

    print("----------------------")

    # Create an alias
    y_pred = rank_labels_ravel_foreground_foreground
    y_true = ground_truth_labels_ravel_foreground_foreground

    print("Number of pred clusters:", len(np.unique(y_pred)))
    print("Number of true clusters:", len(np.unique(y_true)))

    ######################
    ## Calculate scores ##
    ######################
    # Relabel, to check invariance to permutation
    y_pred_re = relabel(y_pred, background=0)
    # print("Sanity check of relabeling =", np.unique(y_pred - y_pred_re))
    # y_true_re = relabel(y_true, background=0)

    over_segmentation_score, completeness_score, over_mixing_score, homogeneity_score, v_score, nmi_score, ami_score, ari_score, fmi_score = calculate_all_scores(
        y_true, y_pred)

    over_segmentation_score_re, completeness_score_re, over_mixing_score_re, homogeneity_score_re, v_score_re, nmi_score_re, ami_score_re, ari_score_re, fmi_score_re = calculate_all_scores(
        y_true, y_pred_re)

    ###########################
    ###########################
    print_metric("over_segmentation_score", over_segmentation_score,
                 over_segmentation_score/over_segmentation_score_re)
    print_metric("completeness_score", completeness_score,
                 completeness_score/completeness_score_re)
    print("----------------------")

    print_metric("over_mixing_score", over_mixing_score,
                 over_mixing_score/over_mixing_score_re)
    print_metric("homogeneity_score", homogeneity_score,
                 homogeneity_score/homogeneity_score_re)
    print("----------------------")

    print_metric("v_score", v_score,
                 v_score/v_score_re)
    print_metric("ami_score", ami_score,
                 ami_score/ami_score_re)
    print_metric("nmi_score", nmi_score,
                 nmi_score/nmi_score_re)
    print("----------------------")

    print_metric("ari_score", ari_score,
                 ari_score/ari_score_re)
    print_metric("fmi_score", fmi_score,
                 fmi_score/fmi_score_re)
