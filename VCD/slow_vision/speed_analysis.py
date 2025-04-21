import os
import time
import argparse
from pathlib import Path
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

def process_images(folder_path, model, text_prompt, box_threshold, text_threshold, save_annotated=False):
    """
    Process all images in the given folder and its subfolders.
    Returns total inference time and image count.
    """
    total_time = 0
    image_count = 0
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Traverse all subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if file is an image
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                
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