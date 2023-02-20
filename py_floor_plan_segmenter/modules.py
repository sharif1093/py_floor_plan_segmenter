import cv2
import numpy as np
from typing import List
from pathlib import Path

from py_floor_plan_segmenter.segment import create_watershed_segment_skimage, crop_background, crop_single_info, dilate_binary, generate_gaussian_seeds_skimage, make_binary, make_invert, remove_small_connected_components, generate_ridge_map_meijering
from py_floor_plan_segmenter.segment import uncrop_background, remap_border

from py_floor_plan_segmenter.over_segment import remap, find_consistent_global_labels, find_lifespans, generate_map_initial_seeds, superpose_map_initial_seeds
from py_floor_plan_segmenter.graph import find_graph_representation
from py_floor_plan_segmenter.merge import merge_nodes_in_place

from py_floor_plan_segmenter.debugging.debugging_factory import debugger
from py_floor_plan_segmenter.map_alignment import find_alignment_angle, rotate_image


def generate_denoised_alone(raw, **config):
    # Create a local binary just to figure out the crop_info
    binary_for_crop = make_binary(
        raw, **config["binary_for_crop/make_binary"])
    crop_info = crop_single_info(binary_for_crop, **config["crop_single_info"])
    # Delete binary_for_crop to free memory
    del binary_for_crop

    # cropped: Crop rank map to ROI
    rank = crop_background(raw, crop_info)
    debugger.add("cropped", rank)

    # binary: Make image into binary, to remove some noise
    rank = make_binary(rank, **config["binary/make_binary"])
    debugger.add("binary", rank)

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
    debugger.add("denoised", rank)

    # rank: Make image into binary
    rank = make_binary(rank, **config["rank/make_binary"])

    return rank, crop_info


# @profile
def compute_labels_list(binary_dilated: np.ndarray, ridges: np.ndarray, sigma_start, sigma_step, max_iter, debug: bool = False, **config):
    # What is inside in this section should be refactored

    sigmas_list = []
    labels_list = []

    nccs_list = []
    segments_list = []

    iter = 0
    ncc = 1
    sigma = sigma_start
    if sigma_step == 0:
        max_iter = 1
    while (ncc > 0) and (iter < max_iter):
        labels = generate_gaussian_seeds_skimage(
            binary_dilated, sigma=sigma, **config["generate_gaussian_seeds_skimage"])

        # Compute ncc (minus borders and background)
        ncc = max(0, len(np.unique(labels)) - 2)

        sigmas_list += [sigma]
        labels_list += [labels]
        if debug:
            # Process segments
            nccs_list += [ncc]

            segments = create_watershed_segment_skimage(
                ridges, labels, **config["create_watershed_segment_skimage"])
            segments_list += [segments]

        sigma += sigma_step
        iter += 1

    return sigmas_list, labels_list, nccs_list, segments_list


def over_segment(ridges, sigmas_list: List, labels_list: List, **config):
    labels_list = [
        remap(map, {255: 0, 0: len(np.unique(map)) - 1}) for map in labels_list]

    mappers = find_consistent_global_labels(sigmas_list, labels_list)
    lifespan = find_lifespans(mappers)

    map_initial_seeds = generate_map_initial_seeds(
        lifespan, mappers, labels_list, **config["generate_map_initial_seeds"])

    superposed_labels = superpose_map_initial_seeds(
        map_initial_seeds, labels_list)
    debugger.add("superposed_labels", superposed_labels)

    over_segmented = create_watershed_segment_skimage(
        ridges, superposed_labels, **config["create_watershed_segment_skimage"])

    border_label = np.max(over_segmented)

    return over_segmented, border_label


def prepare_segments_for_export(map: np.ndarray, crop_info):
    # segments_uncropped
    map = uncrop_background(map, crop_info, background=0)
    # segments_remapped
    map = remap_border(map)
    return map


def export_global_segments_map(output_path: Path, map: np.ndarray, name="global_prior_map.pgm"):
    print("  .. list of all priors:", np.unique(map))
    cv2.imwrite(str(output_path / name), map)


def do_segment(raw, **config):
    # Find alignment angle and rotate
    if config["alignment"]:
        alignment_angle = find_alignment_angle(
            raw, **config["find_alignment_angle"])
        raw = rotate_image(raw, alignment_angle)
    else:
        alignment_angle = 0
    debugger.add("alignment_angle", alignment_angle)

    # Generate denoised map
    # Increasing sigma_start will reduce noise
    rank, crop_info = generate_denoised_alone(raw,
                                              **config["generate_denoised_alone"])
    debugger.add("rank", rank)
    debugger.add("crop_info", crop_info)

    # binary_dilated: Generate dilated rank map
    binary_dilated = dilate_binary(rank, **config["dilate_binary"])
    debugger.add("binary_dilated", binary_dilated)
    # ridges: Generate the ridge map
    ridges = generate_ridge_map_meijering(
        rank, **config["generate_ridge_map_meijering"])
    debugger.add("ridges", ridges)

    sigmas_list, labels_list, nccs_list, segments_list = compute_labels_list(binary_dilated, ridges,
                                                                             debug=debugger.debug or (
                                                                                 config["sigma_step"] == 0),
                                                                             **config["compute_labels_list"])
    debugger.add("sigmas_list", sigmas_list)
    debugger.add("labels_list", labels_list)
    debugger.add("nccs_list", nccs_list)
    debugger.add("segments_list", segments_list)

    # If a range of sigma was not considered, return.
    if config["sigma_step"] == 0:
        return prepare_segments_for_export(segments_list[0], crop_info)

    segments, border_label = over_segment(ridges,
                                          sigmas_list,
                                          labels_list,
                                          **config["over_segment"])
    debugger.add("segments_fine", segments)

    G, common_borders_map, borders = find_graph_representation(
        segments, border_label)
    debugger.add("G1", G)

    G, segments, common_borders_map, borders = merge_nodes_in_place(G,
                                                                    segments,
                                                                    common_borders_map,
                                                                    borders,
                                                                    border_label,
                                                                    **config["merge_nodes_in_place"])
    debugger.add("G2", G)
    debugger.add("segments_merged", segments)

    return prepare_segments_for_export(segments, crop_info)
