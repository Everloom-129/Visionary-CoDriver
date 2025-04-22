#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Get the project root directory=
# Now we can import the roadside analyzer
from VCD.slow_vision.roadside_analyzer import RoadsideAnalyzer

def process_bdd100k():
    """Process the BDD100K dataset with the roadside analyzer"""
    input_dir =  Path("data") / "BDD100K" / "BDD_masks"
    output_dir =  Path("results") / "BDD100K"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = RoadsideAnalyzer({
        'debug': True,
        'box_threshold': 0.25,
        'text_threshold': 0.20
    })
    
    # Process each video directory
    for video_dir_name in os.listdir(input_dir):
        if not video_dir_name.startswith('video_'):
            continue
            
        video_dir = input_dir / video_dir_name
        if not video_dir.is_dir():
            continue
            
        print(f"Processing folder: {video_dir_name}")
        
        # Find the main image file (e.g., video_0382.png)
        main_image = video_dir / f"{video_dir_name}.png"
        
        if not main_image.exists():
            print(f"Warning: Main image not found in {video_dir}")
            continue
            
        # Create output directory for this video
        video_output_dir = output_dir / video_dir_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path for the visualization
        output_path = video_output_dir / f"road_scene_{video_dir_name}.jpg"
        
        print(f"Analyzing image: {main_image}")
        print(f"Output will be saved to: {output_path}")
        
        # Run the analysis
        obj_dict, p_surface_overlaps = analyzer.detect_road_scene(str(main_image), str(output_path))
        
        print(f"Detected {len(obj_dict)} objects in {video_dir_name}")
        print(f"Analysis complete for {video_dir_name}")
        
    print(f"All analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    process_bdd100k() 