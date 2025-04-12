#!/bin/bash

VIDEO_EXT=("mov" "mp4")

# Define the input and output directories
INPUT_DIR="videos/"
OUTPUT_DIR="YOLOX_outputs/car_videos"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Traverse the input directory
for SUBDIR in $(ls -d $INPUT_DIR*/)
do
    # Traverse all files in the current subdirectory
    for VIDEO_FILE in $SUBDIR*
    do
        # Check if the file has one of the supported extensions
        EXT="${VIDEO_FILE##*.}"  # Extract the file extension
        if [[ " ${VIDEO_EXT[@]} " =~ " ${EXT} " ]]; then
            # Extract the video name without extension
            VIDEO_NAME=$(basename "$VIDEO_FILE" ".$EXT")
            
            # Define the output video path
            OUTPUT_VIDEO="$OUTPUT_DIR/car_$VIDEO_NAME.$EXT"
            
            # Check if the output video already exists
            if [[ -e $OUTPUT_VIDEO ]]; then
                echo "already tracked $OUTPUT_VIDEO"
            else
                echo "start tracking $OUTPUT_VIDEO!"
                # Call the process_video_for_mot function (you'll need to replace this with an actual command)
                python process_video_for_mot.py $VIDEO_FILE $OUTPUT_DIR
            fi
        fi
    done
done
