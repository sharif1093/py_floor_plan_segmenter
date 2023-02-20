import numpy as np
import networkx as nx


def extract_cell_and_pad(map: np.ndarray, center):
    center_x = center[0]
    center_y = center[1]

    height = map.shape[0]
    width = map.shape[1]

    # extract cell
    if center_x == 0:
        lower_x = 0
        lower_pad_x = 1
    else:
        lower_x = center_x - 1
        lower_pad_x = 0

    if center_x == width - 1:
        upper_x = width - 1
        upper_pad_x = 1
    else:
        upper_x = center_x + 1
        upper_pad_x = 0

    if center_y == 0:
        lower_y = 0
        lower_pad_y = 1
    else:
        lower_y = center_y - 1
        lower_pad_y = 0

    if center_y == height - 1:
        upper_y = height - 1
        upper_pad_y = 1
    else:
        upper_y = center_y + 1
        upper_pad_y = 0

    padding_matrix = ((lower_pad_y, upper_pad_y), (lower_pad_x, upper_pad_x))
    cell = np.pad(map[lower_y:upper_y+1, lower_x:upper_x+1],
                  padding_matrix, 'constant', constant_values=0)
    return cell


def all_cell_neighbors(cell: np.ndarray, connectivity: int = 8):
    # Returns a list of all labels within the vicinity of center of a 3x3 cell
    #
    #        a   b   c
    #
    #        d   X   e
    #
    #        f   g   h
    #
    # Any label other than (0|X) is considered a neighbor of X.
    # We create a list of all neighbors.
    assert cell.shape == (3, 3)
    assert connectivity == 8

    neighbors = set([])

    for i in range(3):
        for j in range(3):
            if (cell[i, j] != 0) and (cell[i, j] != cell[1, 1]):
                neighbors.add(cell[i, j])
    return neighbors  # It can have up to 5 neighbors!


def extract_common_borders(map: np.ndarray, border_label: int):
    map_borders = np.zeros_like(map)

    borders = {}
    max_borders_label = 0

    height = map.shape[0]
    width = map.shape[1]

    for x in range(width):
        for y in range(height):
            if (map[y, x] == border_label):
                cell = extract_cell_and_pad(map, (x, y))
                neighbors = all_cell_neighbors(cell)
                if len(neighbors) == 2:
                    neighbor = (min(neighbors), max(neighbors))
                    if not neighbor in borders:
                        max_borders_label += 1
                        borders[neighbor] = max_borders_label
                    # Add the label to map_borders
                    map_borders[y, x] = borders[neighbor]

    return map_borders, borders


def extract_region_borders(map: np.ndarray, region: int, border_label: int):
    map_borders = np.zeros_like(map, dtype=np.bool8)

    height = map.shape[0]
    width = map.shape[1]

    for x in range(width):
        for y in range(height):
            if (map[y, x] == border_label):
                cell = extract_cell_and_pad(map, (x, y))
                neighbors = all_cell_neighbors(cell)
                if region in neighbors:
                    # Add the label to map_borders
                    map_borders[y, x] = True
                else:
                    map_borders[y, x] = False
    return map_borders


def exclude_end_points(map: np.ndarray, inverse: bool = False):
    """Makes borders to False

    Args:
        map (np.ndarray): A boolean numpy 2d array.
    """
    result = map.copy()
    height = map.shape[0]
    width = map.shape[1]

    for x in range(width):
        for y in range(height):
            if (map[y, x] == True):
                cell = extract_cell_and_pad(map, (x, y))

                count = 0
                if cell[0, 1]:
                    count += 1
                if cell[1, 0]:
                    count += 1
                if cell[1, 2]:
                    count += 1
                if cell[2, 1]:
                    count += 1

                if not inverse:
                    if count == 1:
                        result[y, x] = False
                else:
                    if count != 1:
                        result[y, x] = False

    return result

# Graph attributes


def get_area(segments_map: np.ndarray, label: int, resolution: float = 0.040):
    return len(segments_map[segments_map == label]) * resolution * resolution


def get_length(borders_map: np.ndarray, label: int, resolution: float = 0.040):
    # Calculates the length of a border as it is
    return len(borders_map[borders_map == label]) * resolution


def get_position(segments_map: np.ndarray, label: int):
    count = len(segments_map[segments_map == label])
    pos = np.argwhere(segments_map == label).sum(0)/count
    return (int(pos[1]), int(pos[0]))


def create_graph(segments, common_borders_map, borders, border_label):
    # Create the graph
    G = nx.Graph()

    exclude_node_list = [0, border_label]
    # Add nodes!
    for label in sorted(np.unique(segments)):
        if label in exclude_node_list:
            continue
        position = get_position(segments, label)
        area = get_area(segments, label)
        G.add_node(label, area=area, position=position)

    # Add edges!
    for edge in borders:
        length = get_length(common_borders_map, borders[edge])
        G.add_edge(edge[0], edge[1], length=length)
    return G


def find_graph_representation(segments, border_label):
    common_borders_map, borders = extract_common_borders(
        segments, border_label)
    G = create_graph(segments, common_borders_map, borders, border_label)
    return G, common_borders_map, borders
