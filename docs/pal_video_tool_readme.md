# PAL Video Processing Tool
- Contributors: Tony Wang
- 09/25/2024
---


This tool is designed for use in the PAL lab to annotate video, particularly for robotics demos.


## Features

- [x] **Object Detection with Text**: Collabrating with `IDEA-Research`, this library use the SOTA Openset Detection Model: `Grounding DINO Series` to detect objects in the video and overlays text annotations. With API access, this tool is light weight, easy to use and solely built for PAL Lab members. 

- [X] **Masking with Grounded SAM**: With API access to `IDEA-Research`, this tool is able to render masks with `Grounding SAM` to specific objects or regions within the video. 

- [x] **Video Masking**: With `Segment Anything 2`, the tool can render masks to specific objects or regions within the video.

- [ ] **Fancy Visualization**: With `Supervision`, more advanced visualization techniques to enhance the presentation of video content, to be developed...

- [x] **Depth Estimation**: With `ml-depth-pro`, the tool can estimate the depth of given image sequence and analyze results.

## Installation

To install the PAL Video Processing Tool, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Everloom-129/pal_video_tool.git
    ```
2. Navigate to the project directory:
    ```sh
    cd pal_video_tool
    ```
3. Install the required dependencies in conda environment:
    ```sh
    conda create -n pal_video_tool python=3.9
    conda activate pal_video_tool
    pip install -r requirements.txt
    <!-- cd idea-research-api # IDEA-Research's SDK repo Removed with fixed PR now -->
    cd ml-depth-pro
    pip install -e .
    ```
4. Obtain API token from [`IDEA-Research`](https://cloud.deepdataspace.com/dashboard/api-quota) and set it in the environment variable:
    **Option 1: Temporarily set the API key**

    This method sets the API key only for the current terminal session.

    ```sh
    export DDS_CLOUDAPI_TEST_TOKEN='YOUR_API_KEY'
    ```

    **Option 2: Permanently add the API key to your `.bashrc` file**

    This method ensures the API key is set every time you open a new terminal session.

      ```sh
      echo "export DDS_CLOUDAPI_TEST_TOKEN='2681bf4c'" >> ~/.bashrc
      ```

    Reload the terminal to apply the changes:

      ```sh
      source ~/.bashrc
      ```

    **Note**: Remember to replace `'YOUR_API_KEY'` with your actual API key.

## Usage

### Grounding DINO for video
To use video processing tool, run the following command:

```sh
python main.py --input <input_video> --output <output_video> --prompts <detection_prompts> #Optional, default output video will be <input_video_name>_pal.mp4
```

### DepthPro for images

#### run DepthPro for a dir of images
```sh
# first set the  input_dir and output_dir in main func
python generate_depth.py 
```
#### Compare depth maps from RGBD and DepthPro
```sh
python generate_depth.py --input_dir <images_dir> --output_dir <your_dir> --image_name <for rename>
# you can also directly modify the main func for quick update
```

Please read functions' docstring for more information.

## Contributing

Welcome suggestions! Please raise any issue / PR if you are interested in it.
- [ ] Add more scripts for detection
- [x] Add more scripts for segmentation
- [ ] Add more scripts for visualization

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


## Dependencies

This project uses `ml-depth-pro` AND `Segment Anything 2` as submodule. After cloning, initialize the submodule:

```sh
git submodule update --init --recursive
```

## Acknowledgments
- DepthPro: [Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)
- DINOX [DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding](https://arxiv.org/abs/2411.14347)
- Segment Anything 2 [Segment Anything 2](https://arxiv.org/abs/2411.14347)

## Contact
For any questions or feedback, please contact the PAL lab team at `tonyw3@seas.upenn.edu`