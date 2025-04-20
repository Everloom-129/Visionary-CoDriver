import cv2
import numpy as np
import os
import sys
import argparse
from collections import defaultdict


def visualize_tracking_results(video_path, person_tracking_file, car_tracking_file, output_path=None):
    """
    Visualize person and vehicle tracking results by drawing bounding boxes, track_ids and speed status on the original video.
    
    Parameters:
        video_path (str): Path to the original video file
        person_tracking_file (str): Path to txt file with person tracking results
        car_tracking_file (str): Path to txt file with vehicle tracking results
        output_path (str): Path for the output video, defaults to adding "_visualized" suffix
    """
    # Create default output path if not specified
    if output_path is None:
        filename, ext = os.path.splitext(video_path)
        output_path = f"{filename}_visualized{ext}"
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Can also use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read person tracking results
    person_tracks_by_frame = defaultdict(list)
    
    try:
        with open(person_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 11:  # Ensure enough fields, including speed flag
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    top = float(parts[2])
                    left = float(parts[3])
                    width_box = float(parts[4])
                    height_box = float(parts[5])
                    speed_flag = int(parts[10])  # Last field is speed flag
                    
                    person_tracks_by_frame[frame_id].append({
                        'track_id': track_id,
                        'bbox': [left, top, width_box, height_box],
                        'speed': 'slow' if speed_flag == 0 else 'fast',
                        'type': 'person'
                    })
    except Exception as e:
        print(f"Error reading person tracking file: {e}")
        cap.release()
        out.release()
        return
    
    print(f"Loaded person tracking data for {len(person_tracks_by_frame)} frames")
    
    # Read vehicle tracking results
    car_tracks_by_frame = defaultdict(list)
    
    try:
        with open(car_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 11:  # Ensure enough fields, including speed flag
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    top = float(parts[2])
                    left = float(parts[3])
                    width_box = float(parts[4])
                    height_box = float(parts[5])
                    speed_flag = int(parts[10])  # Last field is speed flag
                    
                    car_tracks_by_frame[frame_id].append({
                        'track_id': track_id,
                        'bbox': [left, top, width_box, height_box],
                        'speed': 'slow' if speed_flag == 0 else 'fast',
                        'type': 'car'
                    })
    except Exception as e:
        print(f"Error reading vehicle tracking file: {e}")
        cap.release()
        out.release()
        return
    
    print(f"Loaded vehicle tracking data for {len(car_tracks_by_frame)} frames")
    
    # Merge person and vehicle frame data for processing
    all_frames = set(list(person_tracks_by_frame.keys()) + list(car_tracks_by_frame.keys()))
    
    # Assign different colors to different track_ids
    np.random.seed(42)
    person_color_map = {}
    car_color_map = {} 
    
    person_base_color = (0, 255, 0)
    car_base_color = (0, 0, 255) 
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw all person objects on current frame
        if frame_idx in person_tracks_by_frame:
            for track in person_tracks_by_frame[frame_idx]:
                track_id = track['track_id']
                
                # Assign fixed color for track_id, varying from person base color
                if track_id not in person_color_map:
                    person_color_map[track_id] = (
                        int(np.clip(person_base_color[0] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(person_base_color[1] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(person_base_color[2] + np.random.randint(-50, 50), 0, 255))
                    )
                
                color = person_color_map[track_id]
                bbox = track['bbox']
                speed = track['speed']
                obj_type = track['type']
                
                # Draw bounding box
                y, x, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Prepare label text
                label = f"{obj_type} ID:{track_id} ({speed})"
                
                # Choose color for speed label
                speed_color = (0, 255, 0) if speed == 'slow' else (0, 0, 255)  # Green for slow, red for fast
                
                # Draw background rectangle to enhance text visibility
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                
                # Draw object type, ID and speed label
                cv2.putText(
                    frame, 
                    f"{obj_type} ID:{track_id}", 
                    (x, y - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # Show speed below bounding box
                cv2.putText(
                    frame, 
                    speed, 
                    (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    speed_color, 
                    2
                )
        
        # Draw all vehicle objects on current frame
        if frame_idx in car_tracks_by_frame:
            for track in car_tracks_by_frame[frame_idx]:
                track_id = track['track_id']
                
                # Assign fixed color for track_id, varying from vehicle base color
                if track_id not in car_color_map:
                    car_color_map[track_id] = (
                        int(np.clip(car_base_color[0] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(car_base_color[1] + np.random.randint(-50, 50), 0, 255)),
                        int(np.clip(car_base_color[2] + np.random.randint(-50, 50), 0, 255))
                    )
                
                color = car_color_map[track_id]
                bbox = track['bbox']
                speed = track['speed']
                obj_type = track['type']
                
                # Draw bounding box
                y, x, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Prepare label text
                label = f"{obj_type} ID:{track_id} ({speed})"
                
                # Choose color for speed label
                speed_color = (0, 255, 0) if speed == 'slow' else (0, 0, 255)  # Green for slow, red for fast
                
                # Draw background rectangle to enhance text visibility
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                
                # Draw object type, ID and speed label
                cv2.putText(
                    frame, 
                    f"{obj_type} ID:{track_id}", 
                    (x, y - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # Show speed below bounding box
                cv2.putText(
                    frame, 
                    speed, 
                    (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    speed_color, 
                    2
                )
        
        # Display frame index in top left corner
        cv2.putText(
            frame, 
            f"Frame: {frame_idx}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Save processed frame
        out.write(frame)
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processing frame: {frame_idx}/{total_frames}")
            
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Visualization complete. Output video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize person and vehicle tracking results')
    parser.add_argument('video_path', help='Path to original video file')
    parser.add_argument('person_tracking_file', help='Path to txt file with person tracking results')
    parser.add_argument('car_tracking_file', help='Path to txt file with vehicle tracking results')
    parser.add_argument('--output', '-o', help='Output video path')
    
    args = parser.parse_args()
    
    visualize_tracking_results(args.video_path, args.person_tracking_file, args.car_tracking_file, args.output)