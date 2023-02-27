import argparse
from pathlib import Path
import yaml
import time

from py_floor_plan_segmenter.modules import do_segment
from py_floor_plan_segmenter.modules import export_global_segments_map
from py_floor_plan_segmenter.segment import load_gray_map_from_file
from py_floor_plan_segmenter.debugging.debugging_factory import debugger


if __name__ == "__main__":
    ############################################################
    # CLI PARSER #
    ##############
    parser = argparse.ArgumentParser(
        description="CLI interface for the segment map script")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="The path to the input directory.")
    parser.add_argument("-p", "--output-path", type=Path, required=True,
                        help="Output directory root for segmentation results.")
    parser.add_argument("-o", "--out", type=str, default="", required=False,
                        help="Output directory name for segmentation results.")
    parser.add_argument("-c", "--config", type=Path, default=Path(__file__).parent.absolute() / "default.yml", required=False,
                        help="The yaml file including all hyper-parameters.")

    parser.add_argument('--debug', action='store_true', required=False,
                        help="Will created visualization of mid-level results.")
    parser.add_argument('--animate', action='store_true', required=False,
                        help="Will create an animated visualization of the seeds.")
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

        rank_file = args.input / "rank.png"
        if not rank_file.is_file():
            rank_file = args.input / "rank.pgm"
            if not rank_file.is_file():
                print(
                    f"Neither rank.png nor rank.pgm exist in the {args.input}!")
                exit(1)

    print(f"  .. input: {input_path}")

    # Process yaml config file
    if not args.config.is_file():
        print(f"Config file does not exist: {args.config}")
        exit(1)
    else:
        print(f"  .. config: {args.config}")

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    sigma_start = config["compute_labels_list"]["sigma_start"]
    sigma_step = config["compute_labels_list"]["sigma_step"]

    # Process output path
    # We create the output directory ourselves, no problem.
    output_name = args.out if (args.out != "") else base_name
    output_type = f"sigma={sigma_start}" if sigma_step == 0 else f"sigma={sigma_start},{sigma_step}"
    output_path = args.output_path / output_name / output_type

    print(f"  .. output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # 'animate' needs 'debug' mode
    if ((args.animate == True) and (args.debug == False)):
        print("'--animate' argument must be called with the '--debug' argument.")
        exit(1)

    debugger.init(config, args.debug, args.animate)
    debugger.add("is_range", False if sigma_step == 0 else True)
    raw = load_gray_map_from_file(rank_file)
    start = time.time()
    segments = do_segment(raw, **config)
    end = time.time()
    print(f"  .. processing time: {end-start:4.2f}s")
    export_global_segments_map(output_path, segments)
    debugger.generate_reports(output_path)
