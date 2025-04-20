import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D
import os
import argparse

def analyze_mot_speed(file_path, output_dir="output"):
    """
    Analyze speed information from MOT tracking results.
    
    Parameters:
    file_path (str): Path to the MOT tracking results file
    output_dir (str): Directory to save output visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Reading data from {file_path}...")
    
    # Read data
    columns = ['frame_id', 'object_id', 'x', 'y', 'width', 'height', 
               'confidence', 'unused1', 'unused2', 'unused3', 'speed_class']
    data = pd.read_csv(file_path, header=None, names=columns)
    
    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"Total frames: {data['frame_id'].nunique()}")
    print(f"Total objects: {data['object_id'].nunique()}")
    print(f"Speed class distribution: {data['speed_class'].value_counts().to_dict()}")
    
    # Get unique object IDs
    unique_objects = data['object_id'].unique()
    print(f"Number of unique objects detected: {len(unique_objects)}")
    
    print("\nGenerating visualizations...")
    
    # 1. Speed class distribution
    print("\n1. Creating speed class distribution plot...")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='speed_class', data=data, palette='viridis')
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.title('Distribution of Speed Classes')
    plt.xlabel('Speed Class (0: Slow, 1: Fast)')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/1_speed_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Temporal analysis
    print("\n2. Analyzing speed patterns over time...")
    speed_by_frame = data.groupby('frame_id')['speed_class'].mean()
    
    plt.figure(figsize=(15, 6))
    plt.plot(speed_by_frame.index, speed_by_frame.values, '-o', markersize=3)
    plt.title('Average Speed Class Over Time')
    plt.xlabel('Frame ID')
    plt.ylabel('Average Speed Class (0: All Slow, 1: All Fast)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/2_speed_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Object trajectories by speed
    print("\n3. Creating object trajectory visualization...")
    plt.figure(figsize=(15, 10))
    
    # Color map for speed classes
    colors = {0: 'blue', 1: 'red'}
    
    # Plot trajectories
    for obj_id in unique_objects:
        obj_data = data[data['object_id'] == obj_id]
        
        # Skip objects with few detections
        if len(obj_data) < 5:
            continue
            
        # Color points by speed class
        for i in range(len(obj_data) - 1):
            current = obj_data.iloc[i]
            next_point = obj_data.iloc[i + 1]
            
            plt.plot([current['x'], next_point['x']], 
                     [current['y'], next_point['y']], 
                     color=colors[current['speed_class']], 
                     alpha=0.7,
                     linewidth=1.5)
    
    plt.title('Object Trajectories Colored by Speed Class')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Slow (0)'),
        Line2D([0], [0], color='red', lw=2, label='Fast (1)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig(f'{output_dir}/3_object_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Speed class transitions
    print("\n4. Analyzing speed class transitions...")
    # For each object, analyze speed class changes
    speed_transitions = defaultdict(int)
    
    for obj_id in unique_objects:
        obj_data = data[data['object_id'] == obj_id].sort_values('frame_id')
        
        if len(obj_data) < 2:
            continue
            
        # Count transitions
        for i in range(len(obj_data) - 1):
            current_speed = obj_data.iloc[i]['speed_class']
            next_speed = obj_data.iloc[i + 1]['speed_class']
            transition = (current_speed, next_speed)
            speed_transitions[transition] += 1
    
    # Create transition matrix
    transition_matrix = np.zeros((2, 2))
    for (i, j), count in speed_transitions.items():
        transition_matrix[int(i), int(j)] = count
    
    # Normalize by row
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=['Slow (0)', 'Fast (1)'],
                yticklabels=['Slow (0)', 'Fast (1)'])
    plt.title('Speed Class Transition Probabilities')
    plt.xlabel('Next Speed Class')
    plt.ylabel('Current Speed Class')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_speed_transitions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Transition probabilities:")
    print(f"   Slow → Slow: {normalized_matrix[0, 0]:.2f}")
    print(f"   Slow → Fast: {normalized_matrix[0, 1]:.2f}")
    print(f"   Fast → Slow: {normalized_matrix[1, 0]:.2f}")
    print(f"   Fast → Fast: {normalized_matrix[1, 1]:.2f}")
    
    # 5. Speed by object ID
    print("\n5. Analyzing speed class distribution by object...")
    obj_speed_counts = data.groupby('object_id')['speed_class'].value_counts().unstack().fillna(0)
    
    # Calculate fast movement percentage
    if 1.0 in obj_speed_counts.columns:
        obj_speed_counts['fast_percentage'] = obj_speed_counts[1.0] / (obj_speed_counts[0.0] + obj_speed_counts[1.0]) * 100
    else:
        obj_speed_counts['fast_percentage'] = 0
    
    # Sort and take top 20 objects
    sorted_objs = obj_speed_counts.sort_values('fast_percentage', ascending=False)
    top_20_objs = sorted_objs.head(20)
    
    plt.figure(figsize=(15, 6))
    top_20_objs['fast_percentage'].plot(kind='bar')
    plt.title('Percentage of Fast Movement for Top 20 Objects')
    plt.xlabel('Object ID')
    plt.ylabel('Percentage of Fast Movement (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_speed_by_object.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 1.0 in obj_speed_counts.columns:
        print(f"   Average percentage of fast movement across all objects: {obj_speed_counts['fast_percentage'].mean():.2f}%")
    
    # 6. Spatial distribution
    print("\n6. Creating spatial distribution of speed classes...")
    plt.figure(figsize=(15, 10))
    
    # Plot slow points
    slow_data = data[data['speed_class'] == 0]
    plt.scatter(slow_data['x'], slow_data['y'], c='blue', alpha=0.2, s=10, label='Slow (0)')
    
    # Plot fast points
    fast_data = data[data['speed_class'] == 1]
    plt.scatter(fast_data['x'], fast_data['y'], c='red', alpha=0.2, s=10, label='Fast (1)')
    
    plt.title('Spatial Distribution of Speed Classes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/6_spatial_speed_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Individual object speed patterns
    print("\n7. Analyzing speed patterns for individual objects...")
    sample_objects = []
    for obj_id in unique_objects:
        obj_data = data[data['object_id'] == obj_id]
        if len(obj_data) >= 20:  # Only consider objects with enough data points
            sample_objects.append(obj_id)
    
    # Take top 5 objects with most data points
    sample_objects = sorted(sample_objects, 
                          key=lambda x: len(data[data['object_id'] == x]), 
                          reverse=True)[:5]
    
    plt.figure(figsize=(15, 8))
    
    for obj_id in sample_objects:
        obj_data = data[data['object_id'] == obj_id].sort_values('frame_id')
        
        # Create a rolling speed indicator
        plt.plot(obj_data['frame_id'], obj_data['speed_class'], 
                 '-o', markersize=4, label=f'Object {obj_id}')
    
    plt.title('Speed Class Over Time for Selected Objects')
    plt.xlabel('Frame ID')
    plt.ylabel('Speed Class (0: Slow, 1: Fast)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yticks([0, 1], ['Slow', 'Fast'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_object_speed_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Calculate actual speeds
    print("\n8. Calculating and analyzing actual movement speeds...")
    speed_data = []
    
    for obj_id in unique_objects:
        obj_data = data[data['object_id'] == obj_id].sort_values('frame_id')
        
        if len(obj_data) < 5:
            continue
            
        # Calculate speed between consecutive frames
        for i in range(len(obj_data) - 1):
            current = obj_data.iloc[i]
            next_point = obj_data.iloc[i + 1]
            
            # Only consider consecutive frames
            if next_point['frame_id'] == current['frame_id'] + 1:
                dx = next_point['x'] - current['x']
                dy = next_point['y'] - current['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                speed_data.append({
                    'object_id': obj_id,
                    'frame_id': current['frame_id'],
                    'x': current['x'],
                    'y': current['y'],
                    'speed': distance,
                    'speed_class': current['speed_class']
                })
    
    speed_df = pd.DataFrame(speed_data)
    
    if len(speed_df) > 0:
        # Plot speed distribution by class
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='speed_class', y='speed', data=speed_df, palette='viridis')
        plt.title('Distribution of Calculated Speeds by Speed Class')
        plt.xlabel('Speed Class (0: Slow, 1: Fast)')
        plt.ylabel('Speed (pixels per frame)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/8_speed_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate speed statistics
        avg_slow = speed_df[speed_df['speed_class'] == 0]['speed'].mean()
        if 1 in speed_df['speed_class'].values:
            avg_fast = speed_df[speed_df['speed_class'] == 1]['speed'].mean()
            print(f"   Average speed for slow objects: {avg_slow:.2f} pixels/frame")
            print(f"   Average speed for fast objects: {avg_fast:.2f} pixels/frame")
            print(f"   Fast objects move approximately {avg_fast/avg_slow:.2f}x faster than slow objects")
        else:
            print(f"   Average speed for slow objects: {avg_slow:.2f} pixels/frame")
    
        # 9. Global speed trends
        print("\n9. Analyzing global speed trends over time...")
        avg_speeds = speed_df.groupby('frame_id')['speed'].mean()
        
        plt.figure(figsize=(15, 6))
        plt.plot(avg_speeds.index, avg_speeds.values, '-o', markersize=3)
        plt.title('Average Object Speed Over Time')
        plt.xlabel('Frame ID')
        plt.ylabel('Average Speed (pixels per frame)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/9_average_speed_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 10. Spatial speed heatmap
        print("\n10. Creating spatial heatmap of speeds...")
        plt.figure(figsize=(15, 10))
        
        # Create a 2D histogram
        h = plt.hist2d(speed_df['x'], speed_df['y'], weights=speed_df['speed'], 
                     bins=30, cmap='plasma')
        
        plt.colorbar(h[3], label='Average Speed (pixels per frame)')
        plt.title('Spatial Heatmap of Object Speeds')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.savefig(f'{output_dir}/10_speed_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nAnalysis complete! All visualizations have been saved to the output directory.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze MOT speed data.")
    parser.add_argument('file_path', type=str, help='Path to the MOT tracking results file')
    parser.add_argument('--output_dir', '-o', type=str, default='output', help='Directory to save output visualizations')
    args = parser.parse_args()
    
    analyze_mot_speed(args.file_path, args.output_dir)