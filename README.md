# py_floor_plan_segmenter


## Detailed Comparisons with the state

Click on any of the following cases to view the comparison results with the three state-of-the-art methods introduced in [Bormann et al.](https://ieeexplore.ieee.org/document/7487234). From left to right the methods are: (1) ground truth labels, (2) morphologic segmentation, (3) distance-transform-based segmentation, (4) Voronoi-graph-based segmentation, and (5) our down-sampling-based segmentation.

<details><summary>With furniture</summary>
<p>
  <img src="./sandbox/results/furnished.png" alt="Furnished maps benchmark">
</p>
</details>

<details><summary>No furniture</summary>
<p>
  <img src="./sandbox/results/no_furniture.png" alt="No furniture maps benchmark">
</p>
</details>

## Usage

### Interactive testing in the container

```bash
# Build the docker image
docker build -t py_floor_plan_segmenter .
# Running the container in interactive mode
docker run -u cloud -v .:/app -it py_floor_plan_segmenter bash

# Example command to only do the segmentation
python -m py_floor_plan_segmenter -i /app/sandbox/maps/sample_1 -p /app/sandbox/out
# Example command with debug and animate flags
python -m py_floor_plan_segmenter -i /app/sandbox/maps/sample_1 -p /app/sandbox/out --debug --animate
```

### Running the benchmarks

When in docker interactive mode:

```bash
cd /app

./run_benchmark.sh benchmark/no_furniture 
./run_benchmark.sh benchmark/furnished
```

### Evaluation of benchmarks

```bash
export BUILDING=08_lab_f && export TYPE=no_furniture && python -m py_floor_plan_segmenter.evaluate -i sandbox/out/benchmark/$TYPE/$BUILDING/sigma=1.0,0.5 -p sandbox/eval/$TYPE -g sandbox/maps/benchmark/groundtruth/$BUILDING
```

### Running the API server

```bash
# Clone the repository
git clone https://gitlab.tmecosys.net/mohammadreza.sharif/py_room_segmenter
cd py_room_segmenter

docker-compose build --no-cache
docker-compose up -d
```

Then, open your browser at the address: `http://0.0.0.0:8008/docs`
