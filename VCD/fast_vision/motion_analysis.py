import numpy as np
import cv2
from typing import List, Tuple, Union, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_relative_speed(tracked_objects: np.ndarray, 
                           frame_rate: float = 30.0,
                           img_width: int = 1920,
                           img_height: int = 1080) -> Dict[int, float]:
    """Calculate relative speed of tracked objects with respect to ego vehicle.
    
    The method:
    1. Normalizes coordinates by image dimensions to handle different resolutions
    2. Calculates relative motion by comparing object's position change with expected change
       due to ego motion (objects appear to move outward from center as car moves forward)
    3. Uses size change to estimate z-axis motion (approaching/moving away)
    
    Args:
        tracked_objects: Array of shape (N, 6) containing tracking results in MOT format
                        [frame_id, track_id, x, y, w, h]
        frame_rate: Video frame rate (FPS)
        img_width: Video frame width
        img_height: Video frame height
        
    Returns:
        Dictionary mapping track_id to relative speed value
        Positive speed means object is moving faster than ego vehicle
        Negative speed means object is moving slower than ego vehicle
    """
    if len(tracked_objects) < 2:
        return {}
        
    relative_speeds = {}
    track_ids = np.unique(tracked_objects[:, 1])
    
    # Image center for relative motion calculation
    center_x = float(img_width) / 2
    center_y = float(img_height) / 2
    
    for track_id in track_ids:
        # Get trajectory for this track_id
        track_mask = tracked_objects[:, 1] == track_id
        trajectory = tracked_objects[track_mask].astype(np.float64)  # Convert to float
        
        if len(trajectory) < 2:
            continue
            
        # Sort by frame_id
        trajectory = trajectory[trajectory[:, 0].argsort()]
        
        # Calculate normalized center points and sizes
        centers = trajectory[:, 2:4] + trajectory[:, 4:6] / 2
        centers_norm = np.zeros_like(centers)
        centers_norm[:, 0] = centers[:, 0] / img_width
        centers_norm[:, 1] = centers[:, 1] / img_height
        
        sizes = trajectory[:, 4:6]
        sizes_norm = np.zeros_like(sizes, dtype=np.float64)
        sizes_norm[:, 0] = sizes[:, 0] / img_width
        sizes_norm[:, 1] = sizes[:, 1] / img_height
        
        # Calculate relative motion components
        
        # 1. Radial motion (from image center)
        vectors_to_center = centers - np.array([center_x, center_y])
        radial_distances = np.linalg.norm(vectors_to_center, axis=1)
        radial_motion = np.diff(radial_distances)
        
        # 2. Size change (indicates z-axis motion)
        size_changes = np.diff(np.mean(sizes_norm, axis=1))
        
        # 3. Lateral motion (perpendicular to radial)
        lateral_motion = np.zeros_like(radial_motion)
        for i in range(len(centers) - 1):
            # Project motion vector onto perpendicular of radial direction
            motion_vector = centers_norm[i+1] - centers_norm[i]
            radial_dir = vectors_to_center[i] / (radial_distances[i] + 1e-6)  # Add small epsilon to avoid division by zero
            lateral_motion[i] = np.linalg.norm(
                motion_vector - np.dot(motion_vector, radial_dir) * radial_dir
            )
        
        # Combine motion components
        # - Positive radial_motion means object moving away from center (slower than ego)
        # - Positive size_change means object getting larger (faster than ego)
        # - Large lateral_motion indicates crossing motion
        relative_speed = np.mean(
            -radial_motion * 0.5 +  # Weight radial motion negatively
            size_changes * 2.0 +    # Weight size changes more heavily
            lateral_motion * 0.3     # Small weight for lateral motion
        )
        
        relative_speeds[int(track_id)] = float(relative_speed * frame_rate)
    
    return relative_speeds

def classify_relative_speed(speeds: Dict[int, float],
                          threshold: float = 0.05) -> Dict[int, str]:
    """Classify relative speeds into 'slow' or 'fast' categories.
    
    Args:
        speeds: Dictionary mapping track_id to relative speed
        threshold: Speed threshold for classification
        
    Returns:
        Dictionary mapping track_id to speed classification ['slow', 'fast']
    """
    classifications = {}
    
    for track_id, speed in speeds.items():
        # Positive speed means object moving faster than ego vehicle
        classifications[track_id] = 'fast' if speed > threshold else 'slow'
            
    return classifications

def test_motion_analysis():
    """Test the motion analysis functions with synthetic data."""
    # Create synthetic tracking data simulating:
    # - Track 1: Moving away from center (slower than ego)
    # - Track 2: Moving towards center (faster than ego)
    # - Track 3: Crossing motion
    synthetic_tracks = np.array([
        # frame_id, track_id, x, y, w, h
        [0, 1, 800, 500, 50, 50],    # Center-ish
        [1, 1, 750, 450, 45, 45],    # Moving towards edge, getting smaller
        
        [0, 2, 900, 600, 40, 40],    # Right of center
        [1, 2, 850, 550, 50, 50],    # Moving towards center, getting larger
        
        [0, 3, 400, 500, 50, 50],    # Left side
        [1, 3, 500, 500, 50, 50],    # Moving right (crossing)
    ])
    
    # Test relative speed calculation
    speeds = calculate_relative_speed(synthetic_tracks)
    logger.info(f"Calculated relative speeds: {speeds}")
    
    # Test speed classification
    classifications = classify_relative_speed(speeds)
    logger.info(f"Speed classifications: {classifications}")
    
    return speeds, classifications

if __name__ == "__main__":
    test_motion_analysis() 