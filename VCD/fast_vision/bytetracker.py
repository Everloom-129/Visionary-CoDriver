import os
import cv2
import torch
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
from yolox.data.data_augment import preproc
from yolox.utils import postprocess
from VCD.fast_vision.motion_analysis import calculate_relative_speed, classify_relative_speed

# BUG : the model is not correctly loaded 
MODEL_PATH = "config/weights/bytetrack_x_mot17.pth.tar"

class ByteTrackerWrapper:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30, class_ids=[0, 2]):
        """Initialize ByteTracker with YOLOX detector
        
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Number of frames to keep lost tracks 
            match_thresh: Matching threshold for tracking
            frame_rate: Frame rate of input video
            class_ids: List of class IDs to detect. Default: [0: person, 2: car]
        """
        # Get experiment configuration - use MOT configuration
        self.exp = get_exp("config/yolox_x.py", None)
        self.exp.test_size = (800, 1440)  # MOT standard input size
        self.exp.test_conf = 0.1  # Detection confidence threshold
        self.exp.nmsthre = 0.7  # NMS threshold
        self.exp.num_classes = 80  # COCO dataset has 80 classes
        
        # Store class IDs to detect
        self.class_ids = class_ids
        
        # Initialize YOLOX model
        self.model = self.exp.get_model()
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            self.model = fuse_model(self.model)
        
        # Initialize tracker
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
        
        self.timer = Timer()
        self.frame_id = 0
        
    def preprocess(self, img):
        """Preprocess image for YOLOX inference"""
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        # Preprocess image
        img, ratio = preproc(img, self.exp.test_size, self.exp.rgb_means, self.exp.std)
        img_info["ratio"] = ratio
        
        img = torch.from_numpy(img).unsqueeze(0).float()
        if torch.cuda.is_available():
            img = img.cuda()
            
        return img, img_info

    def inference(self, img):
        """Run YOLOX inference"""
        with torch.no_grad():
            outputs = self.model(img)
            if self.model.decoder is not None:
                outputs = self.model.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf,
                self.exp.nmsthre
            )
        return outputs
    
    def track_video(self, video_path, output_path=None, save_result=False):
        """Track objects in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save tracking visualization
            save_result: Whether to save tracking results
        """
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if save_result:
            save_folder = os.path.dirname(output_path)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, os.path.basename(video_path).split('.')[0] + "_mot17.txt")
            
        tracked_bboxes = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_id += 1
            self.timer.tic()
            
            # Preprocess
            img, img_info = self.preprocess(frame)
            
            # Inference
            outputs = self.inference(img)
            
            # Process detections
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                
                # Filter detections by class
                mask = np.isin(outputs[:, 6], self.class_ids)  # class id is in column 6
                outputs = outputs[mask]
                
                if len(outputs) > 0:
                    detections = outputs[:, :5]  # x1,y1,x2,y2,score
                    class_ids = outputs[:, 6]  # Store class IDs for saving
                    
                    # Scale coordinates
                    detections[:, :4] /= img_info["ratio"]
                    
                    # Convert to xywh format for tracker
                    bboxes = detections[:, :4]
                    scores = detections[:, 4]
                    
                    # Update tracker
                    online_targets = self.tracker.update(
                        bboxes, scores, [img_info["height"], img_info["width"]]
                    )
                    
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_classes = []
                    
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        
                        # Find corresponding class ID
                        det_idx = (bboxes == t.tlbr).all(axis=1).nonzero()[0][0]
                        online_classes.append(class_ids[det_idx])
                        
                        if save_result:
                            tracked_bboxes.append([
                                frame_count, tid, tlwh[0], tlwh[1], 
                                tlwh[2], tlwh[3], t.score, 
                                class_ids[det_idx]  # Save class ID
                            ])
                    
                    if save_result:
                        online_im = plot_tracking(
                            img_info["raw_img"],
                            online_tlwhs,
                            online_ids,
                            frame_id=self.frame_id,
                            fps=1.0 / self.timer.average_time
                        )
                        
                        if output_path:
                            cv2.imwrite(
                                os.path.join(save_folder, f"{frame_count:05d}.jpg"),
                                online_im
                            )
            
            frame_count += 1
            self.timer.toc()
            
        # Save tracking results in MOT format with class IDs
        if save_result and len(tracked_bboxes) > 0:
            tracked_bboxes = np.array(tracked_bboxes)
            np.savetxt(
                save_path,
                tracked_bboxes,
                fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d",  # Added format for class ID
                delimiter=","
            )
            
        cap.release()
        return tracked_bboxes

def process_jaad_videos(data_root="data/JAAD", save_root="results/mot"):
    """Process all JAAD videos and generate MOT tracking results with speed analysis"""
    video_dir = os.path.join(data_root, "JAAD_clips")
    os.makedirs(save_root, exist_ok=True)
    
    # Initialize ByteTracker with person and car detection
    tracker = ByteTrackerWrapper(class_ids=[0, 2])  # 0: person, 2: car
    
    # Process each video
    for video_id in range(1, 347):  # JAAD has 346 videos
        video_name = f"video_{video_id:04d}.mp4"
        video_path = os.path.join(video_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
            
        print(f"Processing {video_name}...")
        
        # Track objects in video
        tracked_bboxes = tracker.track_video(
            video_path,
            output_path=os.path.join(save_root, video_name),
            save_result=True
        )
        
        if len(tracked_bboxes) > 0:
            # Calculate speeds for tracked objects
            speeds = calculate_relative_speed(tracked_bboxes, frame_rate=30)
            speed_classes = classify_relative_speed(speeds)
            
            # Save speed analysis results
            speed_results = np.column_stack((tracked_bboxes, speeds, speed_classes))
            np.savetxt(
                os.path.join(save_root, f"{video_name}_speeds.txt"),
                speed_results,
                fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%s",  # Updated format
                delimiter=","
            )

if __name__ == "__main__":
    process_jaad_videos()
