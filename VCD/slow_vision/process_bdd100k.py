#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from VCD.utils.time_utils import run_time_decorator
# Get the project root directory=
# Now we can import the roadside analyzer
from VCD.slow_vision.roadside_analyzer import RoadsideAnalyzer

class RuntimeAnalyzer:
    """Class to track and analyze runtime statistics"""
    def __init__(self):
        self.stats = defaultdict(list)
        self.start_time = None
        self.current_step = None
        
    def start(self, step_name):
        """Start timing a step"""
        if self.current_step is not None:
            self.end()  # End previous step if exists
        self.current_step = step_name
        self.start_time = time.time()
        
    def end(self):
        """End timing current step"""
        if self.current_step is not None and self.start_time is not None:
            duration = time.time() - self.start_time
            self.stats[self.current_step].append(duration)
            self.current_step = None
            self.start_time = None
            
    def get_statistics(self):
        """Get statistics for all steps"""
        stats_summary = {}
        for step, times in self.stats.items():
            if times:
                stats_summary[step] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times)
                }
        return stats_summary
    
    def save_report(self, output_dir):
        """Save runtime analysis report"""
        stats = self.get_statistics()
        report_path = Path(output_dir) / "runtime_analysis.txt"
        
        with open(report_path, 'w') as f:
            f.write("Runtime Analysis Report\n")
            f.write("=====================\n\n")
            
            # Overall statistics
            total_time = sum(step['total'] for step in stats.values())
            f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
            f.write(f"Number of Videos Processed: {len(self.stats['video_processing'])}\n\n")
            
            # Per-step statistics
            f.write("Per-Step Statistics:\n")
            f.write("-------------------\n")
            for step, data in stats.items():
                f.write(f"\n{step}:\n")
                f.write(f"  Count: {data['count']}\n")
                f.write(f"  Total: {data['total']:.2f}s\n")
                f.write(f"  Mean: {data['mean']:.2f}s\n")
                f.write(f"  Std: {data['std']:.2f}s\n")
                f.write(f"  Min: {data['min']:.2f}s\n")
                f.write(f"  Max: {data['max']:.2f}s\n")
            
            # Performance bottlenecks
            f.write("\nPerformance Bottlenecks:\n")
            f.write("----------------------\n")
            bottlenecks = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
            for step, data in bottlenecks[:3]:
                percentage = (data['total'] / total_time) * 100
                f.write(f"{step}: {percentage:.1f}% of total time\n")

@run_time_decorator
def process_bdd100k():
    """Process the BDD100K dataset with the roadside analyzer"""
    input_dir = Path("data") / "BDD100K" / "BDD100" / "100_results"
    output_dir = Path("results") / "BDD100"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize runtime analyzer
    runtime = RuntimeAnalyzer()
    
    # Initialize the analyzer
    runtime.start("analyzer_initialization")
    analyzer = RoadsideAnalyzer({
        'debug': True,
        'box_threshold': 0.25,
        'text_threshold': 0.20
    })
    runtime.end()
    
    # Process each video directory
    for video_dir_name in os.listdir(input_dir):
        runtime.start("video_processing")
            
        video_dir = input_dir / video_dir_name
        if not video_dir.is_dir():
            continue
            
        print(f"Processing folder: {video_dir_name}")
        
        # Find the main image file
        runtime.start("file_lookup")
        main_image = video_dir / f"{video_dir_name}.png"
        if not main_image.exists():
            print(f"Warning: Main image not found in {video_dir}")
            runtime.end()
            continue
        runtime.end()
        
        # Create output directory
        runtime.start("directory_setup")
        video_output_dir = output_dir / video_dir_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = video_output_dir / f"road_scene_{video_dir_name}.jpg"
        runtime.end()
        
        print(f"Analyzing image: {main_image}")
        print(f"Output will be saved to: {output_path}")
        
        # Run the analysis
        runtime.start("scene_analysis")
        obj_dict, p_surface_overlaps = analyzer.detect_road_scene(str(main_image), str(output_path))
        runtime.end()
        
        print(f"Detected {len(obj_dict)} objects in {video_dir_name}")
        print(f"Analysis complete for {video_dir_name}")
        
        runtime.end()  # End video processing
        
    # Save runtime analysis report
    runtime.save_report(output_dir)
    print(f"All analysis complete. Results saved to {output_dir}")
    print("Runtime analysis report saved to runtime_analysis.txt")

if __name__ == "__main__":
    process_bdd100k() 