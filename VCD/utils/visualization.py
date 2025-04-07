import cv2
import numpy as np
import supervision as sv
from typing import List
import os
import matplotlib.pyplot as plt

def is_image_file(filename):
    IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(filename.endswith(extension) for extension in IMAGE_EXT)

def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color \
        else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def display_mask(SAM_masks, image_path, output_dir, DINO_boxes):
    output_path = os.path.join(output_dir, image_path)
    plt.figure(figsize=(16,9))
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')

    for mask in SAM_masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in DINO_boxes:
        show_box(box, plt.gca())

    plt.savefig(output_path)
    plt.close()

class Visualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.mask_annotator = sv.MaskAnnotator(opacity=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    def annotate_frame(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        if not detections:
            return frame
        
        boxes = [d['bbox'] for d in detections]
        masks = [d['mask'] for d in detections]
        labels = [f"{d['category']} {d['score']:.2f}" for d in detections]

        sv_detections = sv.Detections(
            xyxy=np.array(boxes),
            mask=np.array(masks),
            class_id=np.arange(len(detections))
        )

        frame = self.box_annotator.annotate(scene=frame, detections=sv_detections)
        
        for label, box in zip(labels, boxes):
            x1, y1, _, _ = box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def add_trace(self, frame: np.ndarray, detections: List[dict], frame_number: int) -> np.ndarray:
        if detections:
            boxes = [d['bbox'] for d in detections]
            sv_detections = sv.Detections(xyxy=np.array(boxes))
            if sv_detections.xyxy.any():
                frame = self.trace_annotator.annotate(scene=frame, detections=sv_detections)
        return frame

    def add_text_overlay(self, frame: np.ndarray, text: str, position: tuple = (10, 30)) -> np.ndarray:
        text_anchor = sv.Point(x=position[0], y=position[1])
        return sv.draw_text(scene=frame, text=text, text_anchor=text_anchor, text_color=sv.Color.RED)

    def highlight_region(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        return sv.draw_rectangle(scene=frame, rectangle=region, text_color=sv.Color.YELLOW)

    def create_heatmap(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        if not detections:
            return frame
        
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            heatmap[y1:y2, x1:x2] += 1
        
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)