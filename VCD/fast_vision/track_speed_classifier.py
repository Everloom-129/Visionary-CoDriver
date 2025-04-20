import numpy as np
import os
import sys

def process_tracking_file(file_path, threshold=0.1):
    """
    Process tracking file and add speed classification (0 for slow, 1 for fast).
    If speed classification already exists, overwrite the original value.
    
    Parameters:
        file_path (str): Path to the tracking file.
        threshold (float): Threshold for slow/fast classification.
                          Relative movement greater than this threshold will be classified as fast.
    """
    try:
        # Read file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines of data")
        
        # Check if file already has speed information
        has_speed_already = False
        sample_line = lines[0].strip().split(',') if lines else []
        if len(sample_line) >= 11:
            has_speed_already = True
            print(f"File {file_path} already has speed information. Original values will be overwritten.")
        
        # Parse data
        data = []
        for line in lines:
            items = line.strip().split(',')
            if len(items) >= 10:  # Ensure enough elements
                try:
                    frame_id = int(float(items[0]))
                    track_id = int(float(items[1]))
                    top = float(items[2])
                    left = float(items[3])
                    width = float(items[4])
                    height = float(items[5])
                    
                    data.append({
                        'line': line.strip(),
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'top': top,
                        'left': left,
                        'width': width,
                        'height': height,
                        'center_x': left + width/2,
                        'center_y': top + height/2
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()}. Error: {e}")
        
        print(f"Successfully parsed {len(data)} tracking records")
        
        # Group by track_id
        tracks = {}
        for item in data:
            if item['track_id'] not in tracks:
                tracks[item['track_id']] = []
            tracks[item['track_id']].append(item)
        
        # Sort each track by frame_id
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        print(f"Data grouped into {len(tracks)} unique tracks")
        
        # Calculate speed for each record
        speed_classification = {}
        for track_id, track_data in tracks.items():
            for i in range(len(track_data)):
                if i == 0:  # First frame of this track
                    # No previous frame to compare, so classify as slow
                    speed_classification[(track_data[i]['frame_id'], track_id)] = 0
                else:
                    current = track_data[i]
                    previous = track_data[i-1]
                    
                    # Calculate movement distance
                    dx = current['center_x'] - previous['center_x']
                    dy = current['center_y'] - previous['center_y']
                    
                    # Calculate relative movement (normalized by object size)
                    object_size = (current['width'] + current['height']) / 2
                    if object_size > 0:
                        relative_movement = np.sqrt(dx**2 + dy**2) / object_size
                    else:
                        relative_movement = 0
                    
                    # Classify as slow or fast
                    if relative_movement > threshold:
                        speed_classification[(current['frame_id'], track_id)] = 1  # Fast
                    else:
                        speed_classification[(current['frame_id'], track_id)] = 0  # Slow
        
        # Create new lines with speed classification
        new_lines = []
        for line in lines:
            items = line.strip().split(',')
            if len(items) >= 10:
                try:
                    frame_id = int(float(items[0]))
                    track_id = int(float(items[1]))
                    
                    # Get speed classification
                    speed = speed_classification.get((frame_id, track_id), 0)
                    
                    # If the line already has speed info, replace it
                    if has_speed_already:
                        new_line = ','.join(items[:10]) + ',' + str(speed) + '\n'
                    else:
                        new_line = line.strip() + ',' + str(speed) + '\n'
                    new_lines.append(new_line)
                except (ValueError, IndexError):
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Write to original file
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Processing complete. Speed classification added/updated in {file_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def get_all_files_in_directory(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <tracking_file_dir> [threshold]")
        sys.exit(1)
    
    file_dir = sys.argv[1]

    from tqdm import tqdm
    for file_path in tqdm(get_all_files_in_directory(file_dir)):
        
        if len(sys.argv) >= 3:
            threshold = float(sys.argv[2])
            process_tracking_file(file_path, threshold)
        else:
            process_tracking_file(file_path)  # Use default threshold 0.1