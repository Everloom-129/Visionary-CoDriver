import os
import time
import argparse
from pathlib import Path
import cv2
import re
from groundingdino.util.inference import load_model, load_image, predict, annotate

def process_images(folder_path, model, text_prompt, box_threshold, text_threshold, save_annotated=False):
    """
    Process images in the given folder and its subfolders.
    For each folder, treat images as a video sequence at 30fps and sample at 2Hz.
    Returns total inference time and image count.
    """
    total_time = 0
    image_count = 0
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Process each subfolder separately
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name)
            
            # Get all images in this subfolder
            image_files = []
            for file in os.listdir(subdir_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file)
            
            # Sort images to ensure sequential order
            # Assuming filenames contain frame numbers
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
            
            # Sample at 2Hz from 30fps - keep every 15th frame
            sampled_images = image_files[::15]
            
            print(f"Folder {subdir_path}: {len(image_files)} total frames, {len(sampled_images)} frames after sampling at 2Hz")
            
            # Process sampled images
            for file in sampled_images:
                image_path = os.path.join(subdir_path, file)
                
                try:
                    # Load image
                    image_source, image = load_image(image_path)
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    # Run inference
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold
                    )
                    
                    # Calculate inference time
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    image_count += 1
                    
                    # Print progress
                    print(f"Processed {image_path}, inference time: {inference_time:.4f}s")
                    
                    # Save annotated image if requested
                    if save_annotated:
                        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                        save_path = os.path.join(os.path.dirname(image_path), 
                                              "annotated_" + os.path.basename(image_path))
                        cv2.imwrite(save_path, annotated_frame)
                
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    return total_time, image_count

def main():
    parser = argparse.ArgumentParser(description="Run GroundingDINO on images in folder and subfolders")
    parser.add_argument("folder_path", type=str, help="Path to folder containing images")
    parser.add_argument("--config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                        help="Path to GroundingDINO config file")
    parser.add_argument("--weights", type=str, default="/root/Visionary-CoDriver/config/weights/groundingdino_swint_ogc.pth",
                        help="Path to GroundingDINO weights file")
    parser.add_argument("--text_prompt", type=str, default="chair . person . dog .",
                        help="Text prompt for detection")
    parser.add_argument("--box_threshold", type=float, default=0.35,
                        help="Box threshold for detection")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Text threshold for detection")
    parser.add_argument("--save_annotated", action="store_true",
                        help="Save annotated images")
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        return
    
    print(f"Loading GroundingDINO model from {args.config} with weights {args.weights}")
    # Load model
    model = load_model(args.config, args.weights)
    
    print(f"Processing images in {args.folder_path} and subfolders")
    # Process images
    total_time, image_count = process_images(
        args.folder_path,
        model,
        args.text_prompt,
        args.box_threshold,
        args.text_threshold,
        args.save_annotated
    )
    
    # Calculate and print average inference time
    if image_count > 0:
        avg_time = total_time / image_count
        print(f"\nResults:")
        print(f"Total images processed: {image_count}")
        print(f"Total inference time: {total_time:.2f} seconds")
        print(f"Average inference time per image: {avg_time:.4f} seconds")
    else:
        print("No images processed")

if __name__ == "__main__":
    main()