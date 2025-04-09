import os
import sys
import numpy as np
import json
import torch
import cv2
from PIL import Image
import supervision as sv
from pathlib import Path

# Add paths for the required models
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

config_file = "VCD/slow_vision/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "VCD/slow_vision/GroundingDINO/weights/groundingdino_swint_ogc.pth"
sam_checkpoint = "VCD/slow_vision/GroundingDINO/weights/sam_vit_h_4b8939.pth"
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything Model
from segment_anything import (
    sam_model_registry,
    SamPredictor
)

class GSAMDetector:
    def __init__(
        self,
        config_file,
        grounded_checkpoint,
        sam_checkpoint,
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        bert_base_uncased_path=None
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        # Load Grounding DINO model
        self.grounding_dino_model = self._load_grounding_dino_model(
            config_file, 
            grounded_checkpoint, 
            bert_base_uncased_path
        )
        
        # Initialize SAM predictor
        self.predictor = SamPredictor(
            sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
        )
    
    def _load_image(self, image_path):
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        
        # Transform for Grounding DINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image_tensor

    def _load_grounding_dino_model(self, model_config_path, model_checkpoint_path, bert_base_uncased_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        if bert_base_uncased_path:
            args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def _get_grounding_output(self, image_tensor, text_prompt):
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        # Move model and image to device
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # Run model inference
        with torch.no_grad():
            outputs = self.grounding_dino_model(image_tensor[None], captions=[caption])
        
        # Process outputs
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # Get phrases
        tokenlizer = self.grounding_dino_model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, 
                tokenized, 
                tokenlizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        
        return boxes_filt, pred_phrases

    def detect(self, image_path, text_prompt):
        """
        Perform object detection and segmentation using Grounding DINO and SAM.
        
        Args:
            image_path: Path to the input image
            text_prompt: Text prompt describing objects to detect
            
        Returns:
            Dict containing detection results:
                - boxes: Bounding boxes in xyxy format
                - masks: Segmentation masks
                - labels: Class labels with confidence scores
                - image: Original image
        """
        # Load and process image
        image_pil, image_tensor = self._load_image(image_path)
        original_size = image_pil.size  # (width, height)
        
        # Get bounding boxes from Grounding DINO
        boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
        
        # Scale boxes to image size
        H, W = original_size[1], original_size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        # Convert to xyxy format
        boxes_xyxy = boxes_filt.cpu()
        
        # Load image for SAM
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks with SAM
        self.predictor.set_image(image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_xyxy, image.shape[:2]
        ).to(self.device)
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Extract class names and confidences
        class_names = []
        confidences = []
        
        for phrase in pred_phrases:
            class_name, conf_str = phrase.split('(')
            confidence = float(conf_str[:-1])  # remove the ')'
            class_names.append(class_name.strip())
            confidences.append(confidence)
        
        # Convert to numpy arrays
        boxes_np = boxes_xyxy.numpy()
        masks_np = masks.cpu().numpy()[:, 0, :, :]  # Remove batch dimension
        
        return {
            "image": image,
            "boxes": boxes_np,
            "masks": masks_np,
            "class_names": class_names,
            "confidences": confidences,
            "labels": pred_phrases
        }
    def visualize_results(self, results, output_dir, image_name):
        """Visualize detection results and save them to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        image = results["image"]
        boxes = results["boxes"]
        masks = results["masks"]
        class_names = results["class_names"]
        confidences = results["confidences"]
        
        # Create detections object
        class_ids = np.array([i for i, _ in enumerate(class_names)])
        labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks, 
            class_id=class_ids,
            confidence=np.array(confidences)
        )
        
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        
        # Generate visualizations
        box_annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        box_annotated = label_annotator.annotate(scene=box_annotated, detections=detections, labels=labels)
        mask_annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
        combined = mask_annotator.annotate(scene=box_annotated.copy(), detections=detections)
        
        # Save all visualizations
        for suffix, img in [
            ("original", image),
            ("boxes", box_annotated),
            ("masks", mask_annotated), 
            ("combined", combined)
        ]:
            cv2.imwrite(
                os.path.join(output_dir, f"{image_name}_{suffix}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            print(f"Visualizations saved to {output_dir}/{image_name}_{suffix}.jpg")
        
        return boxes, masks

# Example usage
if __name__ == "__main__":
    # Default model paths
    
    # Create detector
    detector = GSAMDetector(
        config_file=config_file,
        grounded_checkpoint=grounded_checkpoint,
        sam_checkpoint=sam_checkpoint,
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    # Example detection
    image_path = "data/JAAD/images/video_0001/00000.png"
    text_prompt = "car. person. "
    output_dir = "results/JAAD"
    
    # Run detection
    results = detector.detect(image_path, text_prompt)
    
    # Visualize and save results
    image_name = Path(image_path).stem
    detector.visualize_results(results, output_dir, image_name)
