import os
import numpy as np
import cv2
import supervision as sv
from pathlib import Path

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

API_TOKEN = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]

class GroundingDINO:
    def __init__(self, config_path=None):
        """Initialize DINOX client with API token"""
        # Try to load token from config file
        token = self._load_token(config_path)
        self.config = Config(token)
        self.client = Client(self.config)
        
    def _load_token(self, config_path):
        """Load API token from config file or environment variable"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return f.read().strip()
        
        # Fallback to environment variable
        token = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
        if not token:
            raise ValueError("API Token not found. Please set DDS_CLOUDAPI_TEST_TOKEN environment variable or provide a config file.")
        return token
        
    def get_dinox(self, image_path, input_prompts=None):
        """
        Get detection and segmentation results using DINOX API
        
        Args:
            image_path: Path to input image
            input_prompts: Text prompts for detection, if None uses open-vocabulary detection
            
        Returns:
            List of detected objects with bounding boxes and masks
        """
        # Upload image or use URL
        if image_path.startswith(('http://', 'https://')):
            image_url = image_path
        else:
            image_url = self.client.upload_file(image_path)
        
        # Prepare prompt
        text_prompt = "<prompt_free>" if input_prompts is None else input_prompts
        
        # Create task with V2 API
        task = V2Task(api_path="/v2/task/dinox/detection", api_body={
            "model": "DINO-X-1.0",
            "image": image_url,
            "prompt": {
                "type": "text",
                "text": text_prompt
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8
        })
        
        # Set longer timeout for API request
        task.set_request_timeout(15)
        
        # Run task
        self.client.run_task(task)
        
        # Check if we have a valid result
        if not task.result or 'data' not in task.result or 'result' not in task.result['data']:
            print("No valid result from API")
            return []
        
        # Return objects from result
        return task.result['data']['result'].get('objects', [])
    
    def visualize_bbox_and_mask(self, predictions, img_path, output_dir, img_name):
        """
        Visualize detection results with bounding boxes and masks
        
        Args:
            predictions: List of prediction objects from DINOX API
            img_path: Path to input image
            output_dir: Directory to save output images
            img_name: Base name for output images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read input image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None, None
        
        if not predictions or len(predictions) == 0:
            print(f"No objects detected in {img_path}")
            return None, None
        
        # Extract classes, boxes, and masks from predictions
        classes = []
        boxes = []
        masks = []
        confidences = []
        
        for pred in predictions:
            # Extract caption/class
            caption = pred.get("caption", "unknown").lower().strip()
            classes.append(caption)
            
            # Extract bounding box (region)
            region = pred.get("region", [0, 0, 10, 10])
            boxes.append(region)
            
            # For now, create dummy masks since actual mask format is unknown
            h, w = img.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = map(int, region)
            dummy_mask[y1:y2, x1:x2] = True  # Simple box-shaped mask
            masks.append(dummy_mask)
            
            # Use a default confidence if not available
            confidences.append(1.0)  # Default confidence
        
        # If no valid boxes, return
        if not boxes:
            print(f"No valid bounding boxes found in {img_path}")
            return None, None
        
        # Convert to numpy arrays
        boxes_np = np.array(boxes)
        masks_np = np.array(masks)
        
        # Create unique class IDs
        unique_classes = list(set(classes))
        class_id_map = {cls: i for i, cls in enumerate(unique_classes)}
        class_ids = np.array([class_id_map[cls] for cls in classes])
        
        # Create labels
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(classes, confidences)
        ]
        
        # Create detections object for visualization
        detections = sv.Detections(
            xyxy=boxes_np,
            mask=masks_np,
            class_id=class_ids,
        )
        
        # Draw bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        
        # Draw labels
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Save bbox visualization
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox.jpg"), annotated_frame)
        
        # Draw masks
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
        
        # Save mask visualization
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask.jpg"), annotated_frame)
        
        print(f"Annotated image {img_path} has been saved to {output_dir}")
        return boxes, masks
    
    def _rle_to_mask(self, rle_data):
        """
        Convert RLE encoded mask to binary mask
        This is a placeholder - implement based on actual response format
        """
        # The implementation will depend on how masks are encoded in the API response
        # For now, return a dummy mask
        return np.ones((100, 100), dtype=bool)



if __name__ == "__main__":
    dinox = GroundingDINO()
    input_image = "data/JAAD/images/video_0001/00000.png"
    output_dir = "results/JAAD"
    image_file = "00000.png"
    predictions = dinox.get_dinox(input_image, "person")
    dinox.visualize_bbox_and_mask(predictions, input_image, output_dir, os.path.splitext(image_file)[0])

# if __name__ == "__main__":
#     import sys
#     from tqdm import tqdm
    
#     if len(sys.argv) != 3:
#         print("Usage: python dinox_detector.py <input_dir> <output_dir>")
#         sys.exit(1)
        
#     input_dir = sys.argv[1]
#     output_dir = sys.argv[2]
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create config directory and write token file if needed
#     config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
#     os.makedirs(config_dir, exist_ok=True)
    
#     # Check if token is set in environment
#     token = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
#     if token:
#         # Write token to config file for future use
#         with open(os.path.join(config_dir, "dinox_token.txt"), "w") as f:
#             f.write(token)
    
#     # Initialize DINOX client
#     dinox = GroundingDINO(os.path.join(config_dir, "dinox_token.txt"))
    
#     # Process all jpg files in input directory with progress bar
#     for image_file in tqdm(os.listdir(input_dir), desc="Processing images"):
#         if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             input_image = os.path.join(input_dir, image_file)
#             print(f"Processing image: {input_image}")
            
#             # Run prompt-free detection
#             predictions = dinox.get_dinox(input_image)
            
#             # Visualize results
#             dinox.visualize_bbox_and_mask(
#                 predictions,
#                 input_image,
#                 output_dir,
#                 os.path.splitext(image_file)[0]
#             )

#     print(f"Results saved to {output_dir}")

