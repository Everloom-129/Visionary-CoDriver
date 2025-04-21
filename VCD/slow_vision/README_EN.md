## Slow Vision: GroundingDINO Speed Analysis on JAAD

### Setup

1. Install dependencies:
   ```bash
   # Fetch GroundingDINO repo
   git submodule update --remote VCD/slow_vision/GroundingDINO
   cd VCD/slow_vision/GroundingDINO
   pip install -e .
   cd ../../..
   ```

2. Download pre-trained weights:
   ```bash
   mkdir -p config/weights
   # Download GroundingDINO weights
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```

### Usage Guide

Script `VCD/slow_vision/speed_analysis.py` is used for speed analysis on JAAD dataset. Run this script with the following command:

```bash
python VCD/slow_vision/speed_analysis.py <your_path_to_JAAD_dataset>/images --text_prompt person
```

### Result

| Model           | Speed on RTX3090 (FPS)    |
| -------------- | ----------------------------- |
| GroundingDINO | 8.69 |

It can be concluded that GroundingDINO is capable of achieving real-time inference on commercial GPUs.