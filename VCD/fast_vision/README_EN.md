## Fast Vision: Road Scene Perception Module

This module implements an object detection and tracking system using YOLOX+ByteTrack for Multiple Object Tracking (MOT), with additional speed classification functionality.

### Setup

1. Install dependencies:
   ```bash
   # Clone the repository as well as fetching all submodules (if you haven't already)
   git clone --recurse-submodules https://github.com/your-repo/VCD.git
   cd VCD

   # Install ByteTrack
   cd fast_vision/ByteTrack
   pip install -r requirements.txt
   pip install -e .
   cd ../..
   ```

2. Download pre-trained weights:
   ```bash
   mkdir -p config/weights
   # Download YOLOX-X weights
   wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth -O config/weights/yolox_x.pt
   ```

### Usage Guide

#### 1. Run YOLOX+ByteTrack for MOT Inference

```bash
python VCD/fast_vision/bytetracker.py video --path <path_to_video_dataset> -f VCD/fast_vision/ByteTrack/exps/default/yolox_x.py -c config/weights/yolox_x.pt --save_result
```

Example:
```bash
python VCD/fast_vision/bytetracker.py video --path data/JAAD/JAAD_clips/ -f VCD/fast_vision/ByteTrack/exps/default/yolox_x.py -c ./config/weights/yolox_x.pt --save_result
```

The tracking results will be saved as text files in the `./YOLOX_outputs` directory. For each video clip, there will be two .txt files, one showing the tracking results for 'person' and the other for 'car'.

#### 2. Calculate Movement Speed for Each Tracked Object

```bash
python VCD/fast_vision/track_speed_classifier.py <path_to_MOT_results_directory> [threshold]
```

This command will read all text files in the specified directory and classify each object as either 'fast' (marked as 1) or 'slow' (marked as 0) based on the provided threshold. Results are appended to each line in the original MOT text files.

#### 3. Visualize Results

```bash
python VCD/fast_vision/visualize_tracking.py <path_to_video> <path_to_person_MOT_txt_file> <path_to_car_MOT_txt_file> -o <output_path>
```

This will generate a visualization of the tracking results overlaid on the original video.

### Output Format

The MOT results are saved in a standard format where each line represents:
`<frame_id>,<object_id>,<x>,<y>,<width>,<height>,<confidence>,-1,-1,-1,<speed_class>`

- `speed_class`: 0 for slow movement, 1 for fast movement
- `-1`: This value has no specific meaning and is not used. It is reserved solely to maintain the standard output format of Bytetrack.

