# py_floor_plan_segmenter

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
export BUILDING=08_lab_f && python -m py_floor_plan_segmenter.evaluate -i sandbox/out/benchmark/no_furniture/$BUILDING/sigma=1.0,0.5 -g sandbox/maps/benchmark/groundtruth/$BUILDING
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
