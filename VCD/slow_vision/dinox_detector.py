import os
import numpy as np
import cv2
import supervision as sv
from pathlib import Path

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

class DINOX:
    def __init__(self):
        token = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
        if not token:
            raise ValueError("API Token not found. Please set DDS_CLOUDAPI_TEST_TOKEN environment variable.")
        self.config = Config(token)
        self.client = Client(self.config)
        
        
    def get_dinox(self, image_path, input_prompts=None):
        if image_path.startswith(('http://', 'https://')):
            infer_image_url = image_path
        else:
            print(f"Uploading image {image_path}")
            infer_image_url = self.client.upload_file(image_path)
        
        text_prompt = "<prompt_free>" if input_prompts is None else input_prompts
        
        task = V2Task(api_path="/v2/task/dinox/detection", api_body={
            "model": "DINO-X-1.0",
            "image": infer_image_url,
            "prompt": {
                "type": "text",
                "text": text_prompt,
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8
        })
        task.set_request_timeout(15)
        self.client.run_task(task)
        
        return task.result.get('objects', []) if task.result else []

    def _rle_to_mask(self, rle_counts, size):
        # BUG ! Find existing method to do this 
        height, width = size
        mask = np.zeros((height, width), dtype=np.uint8)
        counts = [int(x) for x in rle_counts.split()] if isinstance(rle_counts, str) else rle_counts
        
        current_pos = 0
        for i in range(0, len(counts), 2):
            current_pos += counts[i]
            if current_pos < height * width:
                end_pos = min(current_pos + counts[i + 1], height * width)
                mask.flat[current_pos:end_pos] = 1
                current_pos = end_pos
                
        return mask.astype(bool)

    def visualize_bbox_and_mask(self, predictions, img_path, output_dir, img_name):
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(img_path)
        if img is None or not predictions:
            return None, None

        classes, boxes, masks, confidences = [], [], [], []
        color_map = {
            'road': (0, 255, 0),
            'sidewalk': (255, 255, 0),
            'person': (0, 0, 255),
            'car': (255, 0, 0),
            'unknown': (128, 128, 128)
        }


        
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

if __name__ == "__main__":
    dinox = DINOX()
    input_image = "data/JAAD/images/video_0001/00000.png"
    output_dir = "results/JAAD"
    image_file = "00000.png"
    predictions = dinox.get_dinox(input_image,"person.car.road")
    dinox.visualize_bbox_and_mask(predictions, input_image, output_dir, os.path.splitext(image_file)[0])
