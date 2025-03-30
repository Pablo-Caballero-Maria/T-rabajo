import cv2
import numpy as np
from pathlib import Path
import os

from utils.img_utils import (MONTHS, get_false_color_image, load_bands_from_month)

def plot_video():
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Video output path
    video_path = str(output_dir / "monthly_transitions.mp4")
    
    # Parameters
    fps = 60
    transition_duration = 5  # seconds per transition
    transition_frames = fps * transition_duration
    
    # Get image dimensions from first month
    vv_img, vh_img = load_bands_from_month(MONTHS[0], True)
    false_color = get_false_color_image(vv_img, vh_img)
    height, width, channels = false_color.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Load all monthly images
    monthly_images = []
    for month in MONTHS:
        vv_img, vh_img = load_bands_from_month(month, True)
        false_color = get_false_color_image(vv_img, vh_img)
        # Convert from RGB to BGR for OpenCV
        false_color_bgr = cv2.cvtColor(false_color, cv2.COLOR_RGB2BGR)
        monthly_images.append(false_color_bgr)
    
    # Add the first month at the end to create a loop
    monthly_images.append(monthly_images[0])
    
    print("Generating video frames with interpolation...")
    # Generate transitions between each consecutive pair of months
    for i in range(len(monthly_images) - 1):
        start_img = monthly_images[i]
        end_img = monthly_images[i+1]
        
        # Generate interpolated frames
        for frame in range(transition_frames):
            # Calculate interpolation factor (0 to 1)
            t = frame / transition_frames
            alpha = (1 - np.cos(t * np.pi)) / 2  # Smooth easing

            interpolated_frame = cv2.addWeighted(start_img, 1 - alpha, end_img, alpha, 0)
            # Write frame to video
            video_writer.write(interpolated_frame)
        
        # Show progress
        print(f"Processed transition {i+1}/{len(monthly_images)-1}")
    
    # Release the video writer
    video_writer.release()
    
    print(f"Video created successfully at {video_path}")