import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import argparse
import math
import time
from tqdm import tqdm

from VCD.slow_vision.roadside_analyzer import RoadsideAnalyzer
from VCD.slow_vision.depth_util import predict_depth

def calculate_angle(center_x, center_y, camera_intrinsics):
    """
    Estimate person's relative angle
    camera_intrinsics: [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = camera_intrinsics
    
    # Calculate pixel offset from camera center
    dx = center_x - cx
    
    # Calculate angle in radians
    angle_rad = math.atan2(dx, fx)
    
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_distance_level(avg_depth):
    """
    Categorize person's distance based on depth value
    """
    if avg_depth <= 5:
        return 1
    elif avg_depth <= 10:
        return 2
    elif avg_depth <= 15:
        return 3
    else:
        return 4

def calculate_catagory(center_x, center_y, H=1080, W=1920):
    """
    Divide frame into 2x3 regions (0-5, from top-left to bottom-right)
    """
    # Calculate region dimensions
    region_width = W // 3
    region_height = H // 2
    
    # Calculate region index
    region_x = center_x // region_width
    region_y = center_y // region_height
    
    # Ensure indices are within bounds
    region_x = min(region_x, 2)
    region_y = min(region_y, 1)
    
    return int(region_y * 3 + region_x)

def update_timing_plot(timing_data, output_folder):
    """
    Update timing statistics chart
    
    timing_data: Dictionary with video IDs and timing statistics
    output_folder: Output folder path
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    video_ids = list(timing_data.keys())
    total_times = [data['total_time'] for data in timing_data.values()]
    detection_times = [data['detection_time'] for data in timing_data.values()]
    depth_times = [data['depth_time'] for data in timing_data.values()]
    
    # Set bar positions
    x = np.arange(len(video_ids))
    width = 0.25
    
    # Create bar chart
    plt.bar(x - width, detection_times, width, label='Detection Time')
    plt.bar(x, depth_times, width, label='Depth Estimation Time')
    plt.bar(x + width, total_times, width, label='Total Time')
    
    # Add labels and title
    plt.xlabel('Video ID')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time by Video')
    plt.xticks(x, video_ids, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(output_folder, 'timing_statistics.png'), dpi=300)
    plt.close()

def process_video_folder(video_folder, output_folder, camera_intrinsics, downsample_rate=15):
    """
    Process a video folder
    
    video_folder: Path to folder containing video frames
    output_folder: Path to output CSV files
    camera_intrinsics: [fx, fy, cx, cy] Camera parameters
    downsample_rate: Sampling rate (from 30Hz to 2Hz, keep every Nth frame)
    
    Returns:
    output_csv_path: CSV output path
    timing_info: Timing statistics dictionary
    """
    # Initialize analyzer
    analyzer = RoadsideAnalyzer({
        'debug': False,
        'box_threshold': 0.25,
        'text_threshold': 0.20
    })
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get video folder name for output naming
    video_name = os.path.basename(video_folder)
    output_csv_path = os.path.join(output_folder, f"{video_name}.csv")
    
    # Get all frame image files
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Downsample frames
    downsampled_frames = frame_files[::downsample_rate]
    
    # Initialize results list
    results = []
    
    # Initialize timing variables
    total_start_time = time.time()
    total_detection_time = 0
    total_depth_time = 0
    frame_count = 0
    
    print(f"Processing folder: {video_name}")
    print(f"Total frames: {len(frame_files)}, Downsampled frames: {len(downsampled_frames)}")
    
    # Process each frame
    for i, frame_file in enumerate(tqdm(downsampled_frames)):
        frame_id = int(os.path.splitext(frame_file)[0])  # Assume filename is frame number
        frame_path = os.path.join(video_folder, frame_file)
        
        # Scene analysis timing
        detection_start = time.time()
        obj_dict, _ = analyzer.detect_road_scene(frame_path, None)
        detection_end = time.time()
        detection_time = detection_end - detection_start
        total_detection_time += detection_time
        
        # Depth estimation timing
        depth_start = time.time()
        depth_map = predict_depth(
            frame_path, 
            output_path=os.path.join(output_folder, f"{video_name}_{frame_id}_depth.jpg"),
            model_path='config/weights/dpt_large-midas-2f21e586.pt'
        )
        depth_end = time.time()
        depth_time = depth_end - depth_start
        total_depth_time += depth_time
        
        frame_count += 1
        
        # Process each detected person
        for obj_key, obj in obj_dict.items():
            if obj_key.startswith("person"):
                # Extract person ID
                person_id = obj_key.replace("person", "")
                
                # Get bounding box and mask
                x1, y1, x2, y2 = obj.box
                mask = obj.mask
                
                # Ensure mask matches depth map dimensions
                if mask.shape != depth_map.shape:
                    mask = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]))
                    mask = mask.astype(bool)
                
                # Calculate average depth within mask
                masked_depth = depth_map[mask]
                avg_depth = np.mean(masked_depth) if len(masked_depth) > 0 else 0
                
                # Calculate geometric center
                if np.any(mask):
                    y_indices, x_indices = np.where(mask)
                    center_x = np.mean(x_indices)
                    center_y = np.mean(y_indices)
                else:
                    # Use bounding box center if mask is empty
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                
                # Calculate relative angle
                angle = calculate_angle(center_x, center_y, camera_intrinsics)
                
                # Determine distance level
                distance_level = calculate_distance_level(avg_depth)

                # Determine screen region
                catagory = calculate_catagory(center_x, center_y)
                
                # Store results
                results.append({
                    'Frame_id': frame_id,
                    'Person_id': person_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'Average_Depth': avg_depth,
                    'Angle': angle,
                    'Distance_Level': distance_level,
                    'X': center_x,
                    'Y': center_y,
                    'catagory': catagory,
                })
    
    # Calculate total processing time
    total_time = time.time() - total_start_time
    
    # Calculate average times
    avg_detection_time = total_detection_time / frame_count if frame_count > 0 else 0
    avg_depth_time = total_depth_time / frame_count if frame_count > 0 else 0
    avg_total_time_per_frame = total_time / frame_count if frame_count > 0 else 0
    
    # Output timing statistics
    print(f"\n--- {video_name} Timing Statistics ---")
    print(f"Total processing time: {total_time:.2f} sec")
    print(f"Average processing time: {avg_total_time_per_frame:.2f} sec/frame")
    print(f"Average detection time: {avg_detection_time:.2f} sec/frame")
    print(f"Average depth estimation time: {avg_depth_time:.2f} sec/frame")
    
    # Timing statistics dictionary
    timing_info = {
        'total_time': total_time,
        'detection_time': total_detection_time,
        'depth_time': total_depth_time,
        'frame_count': frame_count,
        'avg_detection_time': avg_detection_time,
        'avg_depth_time': avg_depth_time,
        'avg_total_time_per_frame': avg_total_time_per_frame
    }
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
    else:
        print(f"Warning: No people detected in video {video_name}")
    
    return output_csv_path, timing_info

def main():
    parser = argparse.ArgumentParser(description='Process video folders and analyze pedestrians')

    parser.add_argument('videos_folder', type=str, help='Path to folder containing video folders')
    parser.add_argument('output_folder', type=str, help='Path to output CSV files')
    parser.add_argument('--fx', type=float, default=721.5377, help='Camera parameter fx (default: 721.5377)')
    parser.add_argument('--fy', type=float, default=721.5377, help='Camera parameter fy (default: 721.5377)')
    parser.add_argument('--cx', type=float, default=609.5593, help='Camera parameter cx (default: 609.5593)')
    parser.add_argument('--cy', type=float, default=172.854, help='Camera parameter cy (default: 172.854)')
    parser.add_argument('--downsample', type=int, default=15, 
                        help='Downsampling rate: keep every Nth frame (default: 15)')
    parser.add_argument('--timing_output', type=str, default=None,
                        help='Path to timing statistics CSV output (default: same as output_folder)')
    
    args = parser.parse_args()

    camera_intrinsics = [args.fx, args.fy, args.cx, args.cy]
    
    # Set performance analysis output path
    timing_output = args.timing_output if args.timing_output else args.output_folder
    
    print(f"Camera parameters: {camera_intrinsics}")
    print(f"Downsampling rate: {args.downsample}")
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Get all subfolders (each represents a video)
    video_folders = [os.path.join(args.videos_folder, d) for d in os.listdir(args.videos_folder) 
                     if os.path.isdir(os.path.join(args.videos_folder, d))]
    
    print(f"Found {len(video_folders)} video folders")
    
    # Initialize global timing statistics
    all_timing_data = {}
    total_dpt_times = []
    total_detection_times = []
    
    # Process each video folder
    for video_folder in video_folders:
        video_name = os.path.basename(video_folder)
        try:
            _, timing_info = process_video_folder(video_folder, args.output_folder, camera_intrinsics, args.downsample)
            
            # Collect timing data
            all_timing_data[video_name] = {
                'total_time': timing_info['total_time'],
                'detection_time': timing_info['detection_time'],
                'depth_time': timing_info['depth_time']
            }
            
            # Collect DPT and detection time data
            total_dpt_times.extend([timing_info['avg_depth_time']] * timing_info['frame_count'])
            total_detection_times.extend([timing_info['avg_detection_time']] * timing_info['frame_count'])
            
            # Update timing chart after each video
            update_timing_plot(all_timing_data, args.output_folder)
            
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
    
    # Calculate and output global average times
    avg_dpt_time = np.mean(total_dpt_times) if total_dpt_times else 0
    avg_detection_time = np.mean(total_detection_times) if total_detection_times else 0
    
    print("\n--- Global Timing Statistics ---")
    print(f"Average DPT depth estimation time: {avg_dpt_time:.4f} sec/frame")
    print(f"Average object detection time: {avg_detection_time:.4f} sec/frame")
    
    # Save timing statistics to CSV
    timing_df = pd.DataFrame([{
        'video_id': video_id,
        'total_time': data['total_time'],
        'detection_time': data['detection_time'],
        'depth_time': data['depth_time']
    } for video_id, data in all_timing_data.items()])
    
    timing_csv_path = os.path.join(args.output_folder, 'timing_statistics.csv')
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"Timing statistics saved to: {timing_csv_path}")
    
    print("All videos processed!")

if __name__ == "__main__":
    main()