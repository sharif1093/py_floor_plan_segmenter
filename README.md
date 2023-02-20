# py_floor_plan_segmenter

## Usage

### Interactive testing in the container

```bash
# Build the docker image
docker build -t py_floor_plan_segmenter .
# Running the container in interactive mode
docker run -u cloud -v ./sandbox:/app/sandbox -it py_floor_plan_segmenter bash

# Example command to only do the segmentation
python -m py_floor_plan_segmenter -i /app/sandbox/maps/sample -p /app/sandbox/out
# Example command with debug and animate flags
python -m py_floor_plan_segmenter -i /app/sandbox/maps/sample -p /app/sandbox/out --debug --animate
```

### Running the API server

```bash
# Clone the repository
git clone https://gitlab.tmecosys.net/mohammadreza.sharif/py_room_segmenter
cd py_room_segmenter

docker-compose build --no-cache
docker-compose up -d
```
