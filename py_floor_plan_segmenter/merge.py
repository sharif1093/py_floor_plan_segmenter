from copy import deepcopy
from typing import OrderedDict
import numpy as np
import networkx as nx
from sortedcontainers import SortedList

from py_floor_plan_segmenter.graph import exclude_end_points, extract_region_borders


def merge(G: nx.Graph, u, v):
    """Merge two nodes in a graph

    Get the graph, node one, and node two.
    Merge the two nodes.
    Then process the contraction nodes/edges based on the update rules.

    Args:
        G (nx.Graph): _description_
        u (_type_): _description_
        v (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if not v in G.neighbors(u):
        raise ValueError(f"The input nodes '{u}' and '{v}' are not neighbors!")

    M = nx.contracted_nodes(G, u, v, copy=True, self_loops=False)

    # Merge logic for nodes
    for node, data in M.nodes(data=True):
        if 'contraction' in data:
            area = data['area']
            position = np.array(data['position'])

            for c in data['contraction']:
                position = (area * position + data['contraction'][c]['area'] * np.array(data['contraction'][c]['position'])) \
                    / (area + data['contraction'][c]['area'])
                area += data['contraction'][c]['area']

            M.nodes[node]['position'] = tuple(position)
            M.nodes[node]['area'] = area
            del M.nodes[node]['contraction']

    # Merge logic for edges
    for u, v, data in M.edges(data=True):
        edge = (u, v)
        # print(edge, data)
        if 'contraction' in data:
            length = data['length']
            for c in data['contraction']:
                length += data['contraction'][c]['length']

            M.edges[edge]['length'] = length
            del M.edges[edge]['contraction']

    return M


def merge_segments_map_in_place(G: nx.Graph, segments: np.ndarray, common_borders_map: np.ndarray, borders: np.ndarray, u, v, border_label):
    edge = (min(u, v), max(u, v))

    if not edge in borders:
        raise ValueError(f"The input nodes '{u}' and '{v}' are not neighbors!")

    # Update segments map
    # Join areas
    segments[segments == v] = u

    # Common borders become part of the segment
    label = borders[edge]
    segments[exclude_end_points(common_borders_map == label)] = u

    # Update common_borders_map map
    common_borders_map[common_borders_map == label] = 0

    # Update borders
    del borders[edge]

    # Merge neighbors of v into u
    # If v is a neighbor to any of the neighbors of u,
    # their common borders must be marked with the same label
    u_neighbors = set(G.neighbors(u)) - {v}
    v_neighbors = set(G.neighbors(v)) - {u}

    for vn in v_neighbors:
        if vn in u_neighbors:
            # Merge the two contradicting labels
            edge_old = (min(v, vn), max(v, vn))
            edge_new = (min(u, vn), max(u, vn))

            label_old = borders[edge_old]
            label_new = borders[edge_new]

            common_borders_map[common_borders_map == label_old] = label_new

            del borders[edge_old]
        else:
            # Treat as a new boundary for u
            # Replace (v, vn) with (u, vn) in borders
            edge_old = (min(v, vn), max(v, vn))
            edge_new = (min(u, vn), max(u, vn))

            label = borders[edge_old]
            borders[edge_new] = label
            del borders[edge_old]

    return segments, common_borders_map, borders

# Delete a node from graph


def delete_zero_deg_node_from_segments_map_in_place(G: nx.Graph, segments: np.ndarray, u, border_label):
    if G.degree[u] != 0:
        raise ValueError(f"We can only remove a 0-degree node!")

    # First, convert the borders to background
    segments[extract_region_borders(segments, u, border_label)] = 0
    # Remove the region itself
    segments[segments == u] = 0

    return segments


# Clean graph
def clean_small_disconnected_segments_in_place(G: nx.Graph, segments: np.ndarray, border_label, area_threshold):
    # Calculate the total area of nodes
    total_area = 0
    for _, data in G.nodes(data=True):
        total_area += data['area']

    # Cleaning the graph!
    degrees = deepcopy(G.degree)
    for node, degree in degrees:
        area = G.nodes[node]['area']
        if (degree == 0) and (area < max(area_threshold, 0.01 * total_area)):
            # Delete node!
            segments = delete_zero_deg_node_from_segments_map_in_place(
                G, segments, node, border_label)
            G.remove_node(node)

    return G, segments


def create_priory_queue(G: nx.Graph):
    """This function creates a sorted-by-degree dictionary of nodes.

    This function is used to address nodes with lower degree, first.
    This means merging nodes from the leaf side.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        OrderedDict: OrderedDict[Degree: (Area,Node)]
    """
    q = OrderedDict()
    for node, degree in G.degree:
        area = G.nodes[node]['area']
        if degree in q:
            q[degree].add((area, node))
        else:
            q[degree] = SortedList([(area, node)])
    return q


def find_best_edge(G: nx.Graph, u):
    # We assume that degree of node is greater than 0.
    edge_sorted = SortedList()
    for v in G.neighbors(u):
        edge_sorted.add((-G.edges[u, v]['length'], v))
    return edge_sorted[0][1]


def find_best_priority_for_merge(G: nx.Graph, priority: SortedList, area_threshold: float):
    all_degrees = sorted(list(set(priority.keys())))

    for degree in all_degrees:
        if degree == 0:
            continue
        for el in priority[degree]:
            area = el[0]
            node = el[1]

            if area < area_threshold:
                v = find_best_edge(G, node)
                return (node, v)

    return None


def merge_nodes_in_place(G: nx.Graph, segments: np.ndarray, common_borders_map: np.ndarray, borders, border_label, **config):
    while True:
        G, segments = clean_small_disconnected_segments_in_place(
            G, segments, border_label, config["area_threshold"])
        priority = create_priory_queue(G)
        edge = find_best_priority_for_merge(
            G, priority, config["area_threshold"])
        if edge is None:
            break
        u, v = min(edge), max(edge)

        segments, common_borders_map, borders = \
            merge_segments_map_in_place(
                G, segments, common_borders_map, borders, u, v, border_label)
        G = merge(G, u, v)

    return G, segments, common_borders_map, borders
