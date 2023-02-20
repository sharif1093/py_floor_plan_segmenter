import numpy as np
from copy import deepcopy
from typing import List, Dict


def remap(map, remap_dict):
    map_new = deepcopy(map)
    for key in remap_dict:
        map_new[map == key] = remap_dict[key]
    return map_new


def find_consistent_global_labels(sigmas: List, labels_map_list: List):
    """Finds the global consistent labels per frame

    """
    current_global_label = 0

    prev_labels = labels_map_list[-1]
    prev_monitor = np.array([])
    prev_local_to_global_mapper = {}

    mappers = []

    counter = 0
    for index in range(len(sigmas)-2, -1, -1):
        counter += 1

        labels = labels_map_list[index]
        # Find the list of meaningful labels (no background/empty cells)
        monitor = np.unique(labels)
        monitor = list(set(monitor) - {0, np.max(monitor)})

        ##########################
        # Local to Global Mapper #
        ##########################
        # 1. A mapper that maps current frame to the previous frame
        curr_to_prev_mapper = {}
        current_parent_seeds = []
        for m in prev_monitor:
            new_m = int(np.average(labels[prev_labels == m]))
            if not new_m in curr_to_prev_mapper:
                # It's good, so far new_m seems to be unique
                curr_to_prev_mapper[new_m] = [m]
            else:
                # NOTE: This is the moment of death of m, because it is
                #       going to merge into new_m
                curr_to_prev_mapper[new_m] += [m]
                current_parent_seeds += [new_m]

        # 2. Create a global mapper for the current frame
        local_to_global_mapper = {}
        for m in monitor:
            if m in current_parent_seeds:
                # The current seed is a parent and needs to be mapped to 255
                local_to_global_mapper[m] = 255
            elif (m in curr_to_prev_mapper):
                prev = curr_to_prev_mapper[m][0]
                prev_global = prev_local_to_global_mapper[prev]
                local_to_global_mapper[m] = prev_global
            else:
                # NOTE: This is the moment of birth of m, because it did not
                #       exist in the previous frame
                # Emerging label
                current_global_label += 1
                local_to_global_mapper[m] = current_global_label

        mappers = [local_to_global_mapper] + mappers

        # Keep track of old labels/monitor
        prev_labels = deepcopy(labels)
        prev_monitor = deepcopy(monitor)
        prev_local_to_global_mapper = deepcopy(local_to_global_mapper)

    return mappers


def find_lifespans(mappers: List):
    lifespan = {}

    for index in range(len(mappers)-1, -1, -1):
        mapper = mappers[index]
        for m in mapper:
            key = mapper[m]

            if not key in lifespan:
                # Process lifespan end
                lifespan[key] = [index, index]

            # Process lifespan start
            lifespan[key][0] = index

    return lifespan


def generate_map_initial_seeds(lifespan: Dict, mappers: List, labels_map_list: List, method: str = "start", filter_threshold: int = 0, weights: Dict = {}):
    # Create a map with starts of all seeds
    map_initial_seeds = {}
    for global_label in lifespan:
        start = lifespan[global_label][0]
        end = lifespan[global_label][1]

        # If too short span, then discard
        if (end - start < filter_threshold):
            continue

        if method == 'start':
            index = start
        elif method == 'end':
            index = end
        elif method == 'middle':
            index = int((start+end)/2)
        elif method == 'weighted':
            index = int((weights["start"]*start + weights["end"] * end) /
                        (weights["start"] + weights["end"]))
        else:
            raise ValueError(f"Provided method = {method} does not exist!")

        local_label = -1
        for m in mappers[index]:
            if (mappers[index][m] == global_label):
                local_label = m
                break

        labels = labels_map_list[index]
        map_initial_seeds[global_label] = np.uint8(labels == local_label)

    return map_initial_seeds


def superpose_map_initial_seeds(map_initial_seeds: Dict, labels_map_list: List):
    # Sum all labels
    sum_label_maps = np.zeros_like(labels_map_list[0], dtype=np.int32)
    for key in map_initial_seeds:
        if (key != 255):
            sum_label_maps += map_initial_seeds[key] * key

    # Create the seed for background
    sum_label_maps += np.uint8(labels_map_list[0] == 0) * 255

    return sum_label_maps
