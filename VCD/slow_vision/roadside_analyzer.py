import os
import cv2
import numpy as np
import supervision as sv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import sys

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

# Silence specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from VCD.slow_vision.gsam_detector import GSAMDetector, config_file, grounded_checkpoint, sam_checkpoint
from VCD.utils.visualization import Visualizer
from VCD.utils.time_utils import run_time_decorator

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
        # self.detector = GSAMDetector() # TODO switch within config?
        self.config = config or {}
        self.debug = self.config.get('debug', True)
        
        # Initialize GSAMDetector with defaults
        # Pass None instead of paths if they don't exist to avoid warnings
        self.detector = GSAMDetector(
            config_file=config_file,
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            box_threshold=self.config.get('box_threshold', 0.3),
            text_threshold=self.config.get('text_threshold', 0.25)
        )
        
    def detect_road_scene(self, image_path, output_path=None):
        """
        Detect road, sidewalk, car and people in an image
        Returns:
           obj_dict, p_surface_overlaps
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at path {image_path} could not be loaded.")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a dictionary to store all detected objects
        obj_dict = Counter()
        
        # First try a combined prompt for all objects
        all_prompt = "road. sidewalk. person."
        print("\n--- Trying combined detection with single prompt ---")
        all_results = self.detector.detect(image_path, all_prompt)
        
        # Process results if objects were found
        if all_results and len(all_results["boxes"]) > 0:
            for i, (box, mask, class_name, confidence) in enumerate(zip(
                all_results["boxes"], 
                all_results["masks"], 
                all_results["class_names"], 
                all_results["confidences"]
            )):
                object_type = class_name.lower().strip()
                
                # Count objects of each type to assign unique IDs
                type_count = sum(1 for name, obj in obj_dict.items() if obj.object_type == object_type)
                
                # Add to dictionary
                index = f"{object_type}{type_count}"
                obj_dict[index] = LocationInfo(object_type, type_count, box, mask, confidence)
        
        # If not enough objects were found, try individual prompts
        if len(obj_dict) < 2:
            print("\n--- Trying individual detections with separate prompts ---")
            
            # Define what to detect
            road_sidewalk_prompt = "road. sidewalk."
            person_prompt = "person."
            
            # Get road and sidewalk detections
            print("\n--- Detecting roads and sidewalks ---")
            road_results = self.detector.detect(image_path, road_sidewalk_prompt)
            
            # Process road/sidewalk detections if any found
            if road_results and len(road_results["boxes"]) > 0:
                for i, (box, mask, class_name, confidence) in enumerate(zip(
                    road_results["boxes"], 
                    road_results["masks"], 
                    road_results["class_names"], 
                    road_results["confidences"]
                )):
                    object_type = class_name.lower().strip()
                    
                    # Count objects of each type to assign unique IDs
                    type_count = sum(1 for name, obj in obj_dict.items() if obj.object_type == object_type)
                    
                    # Add to dictionary
                    index = f"{object_type}{type_count}"
                    obj_dict[index] = LocationInfo(object_type, type_count, box, mask, confidence)
            else:
                print("No road or sidewalk detected in the image.")
            
            # Get person detections
            print("\n--- Detecting people ---")
            person_results = self.detector.detect(image_path, person_prompt)
            
            # Process person detections if any found
            if person_results and len(person_results["boxes"]) > 0:
                for i, (box, mask, class_name, confidence) in enumerate(zip(
                    person_results["boxes"], 
                    person_results["masks"], 
                    person_results["class_names"], 
                    person_results["confidences"]
                )):
                    object_type = class_name.lower().strip()
                    
                    # Count objects of each type to assign unique IDs
                    type_count = sum(1 for name, obj in obj_dict.items() if obj.object_type == object_type)
                    
                    # Add to dictionary
                    index = f"{object_type}{type_count}"
                    obj_dict[index] = LocationInfo(object_type, type_count, box, mask, confidence)
            else:
                print("No people detected in the image.")
        
        # Check if we have any detections at all
        if not obj_dict:
            print("No objects detected in the image.")
            return obj_dict, []
        
        # Analyze relationships between people and surfaces
        p_surface_overlaps = self.analyze_person_surface_relationships(obj_dict)
        
        # If output path is provided, visualize and save results
        if output_path:
            self.visualize_results(image, obj_dict, output_path)
            self.save_analysis_results(image_path, output_path, p_surface_overlaps, obj_dict)
            self.save_mot_metadata(image_path, output_path, obj_dict, p_surface_overlaps)
        
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
                overlap, lowest_point = self._is_overlap(person.mask, surface.mask)
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
    
    def _is_overlap(self, mask_a, mask_b):
        """
        Check if the bottom part of mask_a (person) overlaps with or is close to mask_b (surface)
        by projecting the feet area downward.
        
        Args:
            mask_a: Binary mask of person
            mask_b: Binary mask of surface (road/sidewalk)
            
        Returns:
            tuple: (bool indicating overlap, y-coordinate of lowest point)
        """
        # Check the masks are binary
        assert np.logical_or(mask_a == 0, mask_a == 1).all(), "Mask A should be binary"
        assert np.logical_or(mask_b == 0, mask_b == 1).all(), "Mask B should be binary"
        
        # Find the lowest true point in mask A (person's feet)
        y_coords, x_coords = np.nonzero(mask_a)
        if len(y_coords) == 0:
            return False, None
            
        lowest_point = np.max(y_coords)
        
        # Define feet area (bottom 20% of person's height)
        person_height = lowest_point - np.min(y_coords)
        feet_height = int(0.2 * person_height)
        feet_start = lowest_point - feet_height
        
        # Create feet mask
        feet_mask = np.zeros_like(mask_a)
        feet_mask[feet_start:lowest_point, :] = mask_a[feet_start:lowest_point, :]
        
        # Project feet downward by a small distance (5% of image height)
        projection_distance = int(0.05 * mask_a.shape[0])
        projected_feet = np.zeros_like(mask_a)
        
        # Calculate the target slice dimensions
        target_height = min(projection_distance, mask_a.shape[0] - lowest_point - 1)
        if target_height <= 0:
            return False, lowest_point
            
        # Ensure the source and target slices have the same height
        source_height = lowest_point - feet_start
        if source_height > target_height:
            # If source is larger, take the bottom portion
            feet_mask = feet_mask[feet_start + (source_height - target_height):lowest_point, :]
        elif source_height < target_height:
            # If source is smaller, pad with zeros
            pad_height = target_height - source_height
            feet_mask = np.pad(feet_mask[feet_start:lowest_point, :], 
                             ((0, pad_height), (0, 0)), 
                             mode='constant')
        
        # Now perform the projection with matching dimensions
        projected_feet[lowest_point:lowest_point + target_height, :] = feet_mask[:target_height, :]
        
        # Check for overlap between projected feet and surface
        overlap = np.logical_and(projected_feet, mask_b)
        
        # Also check if there's a surface very close below the feet
        if not np.any(overlap):
            # Look for surface within a small distance below feet
            search_area = np.zeros_like(mask_a)
            search_area[lowest_point:min(lowest_point + projection_distance, mask_a.shape[0]-1), :] = 1
            nearby_surface = np.logical_and(search_area, mask_b)
            if np.any(nearby_surface):
                return True, lowest_point
        
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
        
        # Check if there are objects to visualize
        if not obj_dict:
            plt.title("No objects detected", fontsize=20, color='red')
            # Save the visualization
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            return
        
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
        Save analysis results to a text file in a structured format for LLM understanding
        
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
            # Scene Summary
            f.write(f"Scene Analysis for {img_name}:\n\n")
            
            # Check if any objects were detected
            if not obj_dict:
                f.write("No objects detected in the image.\n")
                return
            
            # Count objects by type
            surface_count = sum(1 for name, obj in obj_dict.items() if obj.object_type in ['road', 'sidewalk'])
            person_count = sum(1 for name, obj in obj_dict.items() if obj.object_type == 'person')
            
            # Write scene summary
            f.write(f"Scene contains {person_count} people and {surface_count} surfaces.\n\n")
            
            # Write surface information
            f.write("Surfaces:\n")
            for name, obj in obj_dict.items():
                if obj.object_type in ['road', 'sidewalk']:
                    f.write(f"- {obj.object_type} {obj.id}: area={obj.get_area()} pixels\n")
            
            # Write person information with their locations
            f.write("\nPeople and their locations:\n")
            for person, surfaces in p_surface_overlaps:
                if surfaces:
                    # Get the primary surface (the one with largest overlap)
                    primary_surface = max(surfaces, key=lambda s: s.get_area())
                    f.write(f"- Person {person.id}: standing on {primary_surface.object_type} {primary_surface.id}\n")
                else:
                    f.write(f"- Person {person.id}: not on any detected surface\n")
            
            # Write detailed statistics
            f.write("\nDetailed Statistics:\n")
            f.write(f"Total objects: {len(obj_dict)}\n")
            f.write(f"Total surface area: {sum(obj.get_area() for name, obj in obj_dict.items() if obj.object_type in ['road', 'sidewalk'])} pixels\n")
            f.write(f"Total person area: {sum(obj.get_area() for name, obj in obj_dict.items() if obj.object_type == 'person')} pixels\n")
            
            # Write scene description
            f.write("\nScene Description:\n")
            if person_count > 0:
                f.write(f"This scene shows {person_count} people ")
                if surface_count > 0:
                    surface_types = set(obj.object_type for name, obj in obj_dict.items() if obj.object_type in ['road', 'sidewalk'])
                    f.write(f"in a scene with {', '.join(surface_types)}. ")
                    if any(surfaces for _, surfaces in p_surface_overlaps):
                        f.write("Some people are standing on surfaces.")
                    else:
                        f.write("No people are detected to be standing on surfaces.")
                else:
                    f.write("but no surfaces are detected.")
            else:
                f.write("No people are detected in this scene.")

    def save_mot_metadata(self, image_path, output_path, obj_dict, p_surface_overlaps):
        """
        Save object detection results in MOT format for tracking and fusion.
        Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
        
        Args:
            image_path: Path to input image
            output_path: Base path for output
            obj_dict: Dictionary of detected objects
            p_surface_overlaps: List of (person, surfaces) tuples
        """
        # Extract image name and frame number
        img_name = os.path.basename(image_path).split('.')[0]
        frame_num = int(img_name.split('_')[-1]) if img_name.split('_')[-1].isdigit() else 0
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the output MOT file path
        mot_path = os.path.join(output_dir, f"mot_{img_name}.txt")
        
        # Class mapping for MOT format
        class_map = {
            'person': 1,
            'road': 2,
            'sidewalk': 3
        }
        
        with open(mot_path, 'w') as f:
            # Write header with format information
            f.write("# format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>\n")
            f.write("# class mapping: 1=person, 2=road, 3=sidewalk\n")
            
            # Write each object in MOT format
            for name, obj in obj_dict.items():
                # Convert bounding box to MOT format
                x1, y1, x2, y2 = obj.box
                bb_left = x1
                bb_top = y1
                bb_width = x2 - x1
                bb_height = y2 - y1
                
                # Get class ID
                class_id = class_map.get(obj.object_type, 0)
                
                # Calculate visibility (1.0 for surfaces, based on confidence for persons)
                visibility = 1.0 if obj.object_type in ['road', 'sidewalk'] else obj.confidence
                
                # Write MOT format line
                mot_line = f"{frame_num},{obj.id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{obj.confidence:.2f},{class_id},{visibility:.2f}\n"
                f.write(mot_line)
            
            # Write person-surface relationships
            f.write("\n# Person-Surface Relationships\n")
            for person, surfaces in p_surface_overlaps:
                if surfaces:
                    # Get primary surface (largest overlap)
                    primary_surface = max(surfaces, key=lambda s: s.get_area())
                    f.write(f"# Person {person.id} -> {primary_surface.object_type} {primary_surface.id}\n")
                else:
                    f.write(f"# Person {person.id} -> no surface\n")
        
        print(f"MOT metadata saved to {mot_path}")

def main():
    # Initialize with configuration
    analyzer = RoadsideAnalyzer({
        'debug': True,
        'box_threshold': 0.25,  # Lower threshold to increase chance of detection
        'text_threshold': 0.20
    })
    
    # Use command line argument for image path if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image path
        image_path = "/root/autodl-tmp/JAAD/images/video_0001/00000.png"

    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist.")
        sys.exit(1)
        
    # Create output directory based on image name
    img_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.basename(os.path.dirname(image_path))
    output_dir = f"results/JAAD/{img_dir}_{img_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{img_dir}_{img_name}.jpg")
    
    print(f"Analyzing image: {image_path}")
    print(f"Output will be saved to: {output_dir}")
    
    # Run the scene analysis
    obj_dict, p_surface_overlaps = analyzer.detect_road_scene(image_path, output_path)
    print(f"Detected {len(obj_dict)} objects in the image")
    # print(type(obj_dict['person0'].mask))
    
    
    # Print detected relationships
    for person, surfaces in p_surface_overlaps:
        if surfaces:
            surface_str = ', '.join([f"{surface.object_type}" for surface in surfaces])
            print(f"Person {person.id} is on {len(surfaces)} surfaces: {surface_str}")
        else:
            print(f"Person {person.id} is not on any detected surface")
            
        
if __name__ == "__main__":
    main()
    # INPUT_DIR = "data/BDD100K/BDD_masks/"
    # OUTPUT_DIR = "results/BDD100K/"
    
    # for video_dir in os.listdir(INPUT_DIR):
    #     analyze_dir(os.path.join(INPUT_DIR, video_dir), os.path.join(OUTPUT_DIR, video_dir))