import cv2
import numpy as np
from .api_client import APIClient
from .visualization import Visualizer

class VideoProcessor:
    def __init__(self):
        self.api_client = APIClient()
        self.visualizer = Visualizer()
        self.frame_skip = 10 # Process every 10th frame, adjust as needed
        
    def process_video(self, input_path, output_path, prompts):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                print(f"Warning: Empty frame at frame {frame_count}. Skipping.")
                continue
                
                # TODO: better way to set this?
            if frame_count % self.frame_skip == 0:  # 3 fps, but this will cause flickering
                try:
                    cv2.imwrite('temp_frame.jpg', frame) # TODO: remove this
                    result = self.api_client.detect_objects('temp_frame.jpg', prompts) # TODO: remove this
                    detections = self.convert_result_to_detections(result, height, width)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    # If there's an error, we'll just write the original frame
                    pass
            
            
            # if frame is not None and frame.size > 0:
            if detections:
                frame = self.visualizer.annotate_frame(frame, detections)
                # frame = self.visualizer.add_trace(frame, detections, frame_count)
                frame = self.visualizer.add_text_overlay(frame, f"Frame: {frame_count}")
                frame = self.visualizer.create_heatmap(frame, detections)
                # fill the rest of the frames with the last annotated frame
                out.write(frame)
            # else:
            #     print(f"Warning: Unable to write frame {frame_count}. Skipping.")

            frame_count += 1

        cap.release()
        out.release()
        print(f"Processed {frame_count} frames")
        # cv2.destroyAllWindows() # TODO: fix this

    def convert_result_to_detections(self, result, height, width):
        detections = []
        for obj in result.objects:
            mask = self.api_client.rle2rgba(obj.mask)
            mask_np = np.array(mask)
            
            # Convert RGBA to grayscale
            mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGBA2GRAY)
            
            # Resize mask if necessary
            if mask_gray.shape != (height, width):
                mask_gray = cv2.resize(mask_gray, (width, height))
            
            # Ensure mask is boolean
            mask_bool = mask_gray > 0
            
            detections.append({
                'bbox': obj.bbox,
                'mask': mask_bool,
                'category': obj.category,
                'score': obj.score
            })
        return detections