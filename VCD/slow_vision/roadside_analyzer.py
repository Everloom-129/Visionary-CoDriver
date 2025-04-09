import os
import cv2
import numpy as np
import supervision as sv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

from VCD.slow_vision.DINOX_detector import GroundingDINO
from VCD.utils.visualization import Visualizer
from VCD.utils.time_utils import run_time_decorator

@run_time_decorator
class LocationInfo:
    """Store information about detected objects"""
    def __init__(self, object_type, object_id, box, mask, confidence):
        self.object_type = object_type  # ('sidewalk', 'road', or 'person')
        self.id = object_id             # Unique ID within the type
        self.box = box                  # Bounding box in xyxy format
        self.mask = mask                # Binary mask
        self.confidence = confidence    # Detection confidence
        self.distance = None            # Will be populated later
        self.angle = None               # Will be populated later
        
    def get_area(self):
        """Calculate the area of the object based on its mask"""
        return np.sum(self.mask)

@run_time_decorator
class RoadsideAnalyzer:
    """Given an image, detect road, sidewalk, car, and people, and analyze relationships between people and surfaces"""
    
    def __init__(self, config=None):
        self.dinox_client = GroundingDINO()
        self.config = config or {}
        self.debug = True
        
    def detect_road_scene(self, image_path, output_path=None):
        """
        Detect road, sidewalk, car and people in an image
        
        Args:
            image_path: Path to the input image
            output_path: Path to save output visualization and results
            
        Returns:
            dict: Dictionary of detected objects
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at path {image_path} could not be loaded.")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define what to detect
        road_sidewalk_prompt = "road, sidewalk"
        person_prompt = "person"
        
        # Get detections using DINOX API
        road_predictions = self.dinox_client.get_dinox(image_path, road_sidewalk_prompt)
        person_predictions = self.dinox_client.get_dinox(image_path, person_prompt)
        
        # Create a dictionary to store all detected objects
        obj_dict = Counter()
        
        # Process road/sidewalk detections
        for i, obj in enumerate(road_predictions):
            object_type = obj.category.lower().strip()
            box = np.array(obj.bbox)
            mask = self.dinox_client.client.task_factory.detection_task.rle2mask(
                self.dinox_client.client.task_factory.detection_task.string2rle(obj.mask.counts), 
                obj.mask.size
            )
            confidence = obj.score
            
            # Add to dictionary
            index = f"{object_type}{i}"
            obj_dict[index] = LocationInfo(object_type, i, box, mask, confidence)
        
        # Process person detections
        for i, obj in enumerate(person_predictions):
            object_type = "person"
            box = np.array(obj.bbox)
            mask = self.dinox_client.client.task_factory.detection_task.rle2mask(
                self.dinox_client.client.task_factory.detection_task.string2rle(obj.mask.counts), 
                obj.mask.size
            )
            confidence = obj.score
            
            # Add to dictionary
            index = f"{object_type}{i}"
            obj_dict[index] = LocationInfo(object_type, i, box, mask, confidence)
        
        # Analyze relationships between people and surfaces
        p_surface_overlaps = self.analyze_person_surface_relationships(obj_dict)
        
        # If output path is provided, visualize and save results
        if output_path:
            self.visualize_results(image, obj_dict, output_path)
            self.save_analysis_results(image_path, output_path, p_surface_overlaps, obj_dict)
        
        return obj_dict, p_surface_overlaps
    
    def analyze_person_surface_relationships(self, obj_dict):
        """
        Analyze where each person is standing
        
        Args:
            obj_dict: Dictionary of detected objects
            
        Returns:
            list: List of (person, surfaces) tuples
        """
        p_surface_overlaps = []
        
        for name, person in obj_dict.items():
            if person.object_type != "person":
                continue  # We only want to analyze persons
            
            overlaps = []
            for name, surface in obj_dict.items():
                # We only want to analyze surfaces (road or sidewalk)
                if surface.object_type not in ['road', 'sidewalk']:
                    continue
                
                # Check if the person and the surface overlap
                overlap, lowest_point = self.is_overlap(person.mask, surface.mask)
                if overlap:
                    overlaps.append(surface)
            
            p_surface_overlaps.append((person, overlaps))
        
        if self.debug:
            # Print out the analysis results
            for person, surfaces in p_surface_overlaps:
                if surfaces:
                    surface_str = ', '.join([f"{surface.object_type} {surface.id}" for surface in surfaces])
                    print(f"Person {person.id} is on the {surface_str}")
                else:
                    print(f"Person {person.id} is not on any detected surface")
        
        return p_surface_overlaps
    
    def is_overlap(self, mask_a, mask_b):
        """
        Check if the bottom part of mask_a overlaps with mask_b
        
        Args:
            mask_a: Binary mask of first object
            mask_b: Binary mask of second object
            
        Returns:
            tuple: (bool indicating overlap, y-coordinate of lowest point)
        """
        # Check the masks are binary
        assert np.logical_or(mask_a == 0, mask_a == 1).all(), "Mask A should be binary"
        assert np.logical_or(mask_b == 0, mask_b == 1).all(), "Mask B should be binary"
        
        # Find the lowest true point in mask A (person's feet)
        y_coords, _ = np.nonzero(mask_a)
        if len(y_coords) == 0:
            return False, None
            
        lowest_point = np.max(y_coords) - 10  # Adjust for feet position
        
        # Focus on the bottom part of mask A
        mask_a_bottom = mask_a.copy()
        mask_a_bottom[:lowest_point, :] = 0
        
        # Check for overlap
        overlap = np.logical_and(mask_a_bottom, mask_b)
        
        return np.any(overlap), lowest_point
    
    def visualize_results(self, image, obj_dict, output_path):
        """
        Visualize detection and segmentation results
        
        Args:
            image: Input image
            obj_dict: Dictionary of detected objects
            output_path: Path to save visualization
        """
        plt.figure(figsize=(16, 9))
        plt.imshow(image)
        plt.axis('off')
        
        # Generate random colors for different object types
        colors = {
            'road': [0.1, 0.5, 0.1, 0.5],     # Green
            'sidewalk': [0.5, 0.5, 0.1, 0.5], # Yellow
            'person': [0.9, 0.1, 0.1, 0.5]    # Red
        }
        
        # Draw masks and bounding boxes
        for name, obj in obj_dict.items():
            # Draw mask with color based on object type
            color = colors.get(obj.object_type, [np.random.random(), np.random.random(), np.random.random(), 0.5])
            h, w = obj.mask.shape
            mask_image = obj.mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, 4)
            plt.imshow(mask_image, alpha=0.5)
            
            # Draw bounding box
            x0, y0, x1, y1 = obj.box
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, 
                                             edgecolor=color[:3], 
                                             facecolor=(0, 0, 0, 0), 
                                             lw=2))
            
            # Add label
            plt.text(x0, y0-5, f"{obj.object_type} {obj.id} ({obj.confidence:.2f})", 
                    color='white', fontsize=10, 
                    backgroundcolor=(0, 0, 0, 0.5))
        
        # Save the visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    
    def save_analysis_results(self, image_path, output_path, p_surface_overlaps, obj_dict):
        """
        Save analysis results to a text file
        
        Args:
            image_path: Path to input image
            output_path: Base path for output
            p_surface_overlaps: List of (person, surfaces) tuples
            obj_dict: Dictionary of detected objects
        """
        # Extract image name
        img_name = os.path.basename(image_path).split('.')[0]
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the output text file path
        txt_path = os.path.join(output_dir, f"Info_{img_name}.txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"INFO of {img_name}:\n")
            
            # Write information about each detected object
            for name, obj in obj_dict.items():
                f.write(f"{obj.object_type} {obj.id} has area {obj.get_area()} pixels\n")
            
            # Write information about person-surface relationships
            for person, surfaces in p_surface_overlaps:
                if surfaces:
                    surface_str = ', '.join([f"{surface.object_type} {surface.id}" for surface in surfaces])
                    f.write(f"Person {person.id} is on the {surface_str}, bbox is {person.box}\n")
                else:
                    f.write(f"Person {person.id} is not on any detected surface\n")
            
            # Write summary counts
            num_road_sidewalk = sum(1 for name, obj in obj_dict.items() if obj.object_type in ['road', 'sidewalk'])
            num_people = sum(1 for name, obj in obj_dict.items() if obj.object_type == 'person')
            
            f.write(f"Number of detected objects: {len(obj_dict)}\n")
            f.write(f"Number of road/sidewalk: {num_road_sidewalk}\n")
            f.write(f"Number of people: {num_people}\n")
        
        print(f"Analysis results saved to {txt_path}")


# Example usage
if __name__ == "__main__":
    analyzer = RoadsideAnalyzer({'debug': True})
    image_path = "data/JAAD/images/video_0001/00000.png"
    output_path = "results/JAAD/0001_0001.jpg"
    
    obj_dict, p_surface_overlaps = analyzer.detect_road_scene(image_path, output_path)
    print(f"Detected {len(obj_dict)} objects in the image")
    
    # Print detected relationships
    for person, surfaces in p_surface_overlaps:
        if surfaces:
            print(f"Person {person.id} is on {len(surfaces)} surfaces")
