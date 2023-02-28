import flatdict
from pathlib import Path
from copy import deepcopy
import pprint
import yaml
import numpy as np

from py_floor_plan_segmenter.debugging.visualization import visualize_segmentation, visualize_graph, visualize_segments_double, plot_sigmas_vs_ncc, plot_segment_debug, gen_highlighted_seeds, gen_highlighted_segments, plot_hist_and_sliding_average, dump_color_image, close_all_figures, dump_results
from py_floor_plan_segmenter.debugging.animation import FrameStacker, FrameStackerToFile
from collections import OrderedDict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DebuggingFactory(metaclass=Singleton):
    def __init__(self):
        self.collection = flatdict.FlatDict()
        self.debug = False

    def init(self, config, debug: bool = False, animate: bool = False):
        self.config = config
        self.debug = debug
        self.animate = animate
        self.animate_labels = animate

    def add(self, key, value):
        if not self.debug:
            return

        if key in self.collection:
            raise KeyError("Key already exists...")
        else:
            self.collection[key] = deepcopy(value)

    def generate_reports(self, output_path: Path):
        if not self.debug:
            return

        print(f"  .. generating reports to {output_path}")
        # print("  .. keys: ", end="")
        # pprint.pprint(self.collection.keys(), indent=10)

        with open(output_path/"config.yml", "w") as f:
            yaml.dump(self.config, f, Dumper=yaml.SafeDumper, sort_keys=False)

        if self.collection["is_range"]:
            # Dump pure results
            dump_results(output_path,
                         segments=self.collection["segments_merged_by_edge"],
                         cropped=self.collection["cropped"])

            # merge_by_edge.png & merge_by_edge_no_graph.png
            visualize_segmentation(output_path,
                                   G=self.collection["G3"],
                                   segments=self.collection["segments_merged_by_edge"],
                                   cropped=self.collection["cropped"],
                                   name="merge_by_edge")

            # merge_by_node.png & merge_by_node_no_graph.png
            visualize_segmentation(output_path,
                                   G=self.collection["G2"],
                                   segments=self.collection["segments_merged_by_node"],
                                   cropped=self.collection["cropped"],
                                   name="merge_by_node")

            # before_merge.png & before_merge_no_graph.png
            visualize_segmentation(output_path,
                                   G=self.collection["G1"],
                                   segments=self.collection["segments_fine"],
                                   cropped=self.collection["cropped"],
                                   name="before_merge")

            # graph.png
            visualize_graph(output_path,
                            G=self.collection["G1"],
                            segments=self.collection["segments_fine"],
                            name="graph")

            # oversegmented.png
            title = "Oversegmented map"
            visualize_segments_double(output_path,
                                      self.collection["superposed_labels"],
                                      self.collection["segments_fine"],
                                      self.collection["cropped"],
                                      self.collection["binary_dilated"],
                                      title,
                                      name="oversegmented")

            # ncc_vs_sigma.png
            plot_sigmas_vs_ncc(
                self.collection["sigmas_list"], self.collection["nccs_list"], path=output_path)

            close_all_figures()
            # animation of labels/segments
            if self.animate:
                fs = FrameStacker(OrderedDict({"seeds": {"title": "Highlighted Labels"},
                                               "segments": {"title": "Highlighted Segments"}}))
                for index in range(len(self.collection["sigmas_list"])):
                    sigma = self.collection["sigmas_list"][index]
                    ncc = self.collection["nccs_list"][index]
                    labels = self.collection["labels_list"][index]
                    segments = self.collection["segments_list"][index]
                    res = {"seeds": gen_highlighted_seeds(labels, self.collection["binary_dilated"]),
                           "segments": gen_highlighted_segments(segments, self.collection["cropped"])}
                    fs.add(
                        res, {"title": "$\sigma$ = {:05.2F}, NCC = {:02d}".format(sigma, ncc)})
                fs_to_file = FrameStackerToFile(fs, rows=1, cols=2)
                fs_to_file.process(output_path/"animation.mp4")

            close_all_figures()
            if self.animate_labels:
                output_seeds_directory = output_path / "seeds"
                output_seeds_directory.mkdir(exist_ok=True)
                for index in range(len(self.collection["sigmas_list"])):
                    sigma = self.collection["sigmas_list"][index]
                    labels = self.collection["labels_list"][index]
                    labels_colored = gen_highlighted_seeds(
                        labels, self.collection["binary_dilated"])
                    animate_labels_path = output_seeds_directory / \
                        f"seed_{index}.png"
                    dump_color_image(
                        labels_colored, f"$\sigma$ = {sigma:05.2F}", animate_labels_path)
            close_all_figures()

        else:
            # brief.png
            title = "Angle={:04.1f}, Sigma = {:05.2F}, NCC = {:02d}".format(
                self.collection["alignment_angle"],
                self.collection["sigmas_list"][0],
                self.collection["nccs_list"][0])
            visualize_segments_double(output_path,
                                      self.collection["labels_list"][0],
                                      self.collection["segments_list"][0],
                                      self.collection["cropped"],
                                      self.collection["binary_dilated"],
                                      title,
                                      name="brief")

            # debug.png
            plot_segment_debug(output_path, title,
                               unrotated=self.collection["unrotated"],
                               unrotated_processed=self.collection["unrotated_processed"],
                               unrotated_edges=self.collection["unrotated_edges"],
                               unrotated_lines=self.collection["unrotated_lines"],
                               #    alignment_angle=self.collection["alignment_angle"],
                               cropped=self.collection["cropped"],
                               binary=self.collection["binary"],
                               rank=self.collection["rank"],
                               ridges=self.collection["ridges"],
                               seeds=self.collection["labels_list"][0],
                               segments=self.collection["segments_list"][0],
                               binary_dilated=self.collection["binary_dilated"])

            # hist_vs_sliding_average.png
            plot_hist_and_sliding_average(self.collection["angle_histogram"],
                                          self.collection["angle_histogram_sliding_average"],
                                          x=np.linspace(
                                              0, 90, self.config["hist_resolution"]),
                                          path=output_path,
                                          title='Histogram of angles, win={}, max={}'.format(
                1+2 *
                self.config["find_alignment_angle"]["find_best_alignment_angle_from_histogram"]["window_half_size"],
                self.collection["alignment_angle"]))


# We create a global debugging factory
debugger = DebuggingFactory()
