import glob
import os
from pathlib import Path

import bm3d
import cv2
import numpy as np
from PIL import Image

from utils.denoisers import *
from utils.enlargers import *

# Define months in order

MONTHS = [
    "jan",
    "feb",
    "mar",
    "april",
    "may",
    "june",
    "july",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# MONTHS = ["jan"]  # For testing purposes, only January is used


def load_bands_from_month(month, processed):
    # Find the VV and VH images for the month
    image_dir = (
        os.path.join("images_processed", month)
        if processed
        else os.path.join("images_original", month)
    )

    vv_path = glob.glob(os.path.join(image_dir, "*VV_(Raw).png"))
    vh_path = glob.glob(os.path.join(image_dir, "*VH_(Raw).png"))

    # Load the images
    vv_img = np.array(Image.open(vv_path[0]).convert("L"))
    vh_img = np.array(Image.open(vh_path[0]).convert("L"))

    return vv_img, vh_img


def get_false_color_image(vv_img, vh_img):
    # Create a false color image (R=VV, G=VH, B=SDWI), where SDWI is ln(10 * VV * VH) ** 2
    # Avoid division by zero
    epsilon = 1e-10

    # Normalize each channel to 0-255
    vv_norm = (
        (vv_img - np.min(vv_img)) / (np.max(vv_img) - np.min(vv_img) + epsilon) * 255
    )
    vh_norm = (
        (vh_img - np.min(vh_img)) / (np.max(vh_img) - np.min(vh_img) + epsilon) * 255
    )

    sdwi = np.log(200 * vv_img * vh_img + epsilon) ** 2
    sdwi_norm = (sdwi - np.min(sdwi)) / (np.max(sdwi) - np.min(sdwi) + epsilon) * 255

    # Create RGB image
    false_color = np.stack([vv_norm, vh_norm, sdwi_norm], axis=2).astype(np.uint8)

    return false_color


def save_images():
    # Iterate over each month
    for month in MONTHS:
        print(f"Processing month: {month}")

        # Load original VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, False)
        sdwi_norm = get_false_color_image(vv_img, vh_img)[:, :, 2]

        # First denoise the images
        print(f"Applying denoising to images for month: {month}")
        sdwi_norm_denoised = denoise_image_median(sdwi_norm)

        # Enlarge the images
        print(f"Enlarging images for month: {month}")
        sdwi_norm_enlarged = enlarge_image_nearest_neighbor(sdwi_norm_denoised, 2)

        # Save the enlarged images
        output_dir = Path("images_processed", month)
        output_dir.mkdir(parents=True, exist_ok=True)

        sdwi_norm_enlarged.save(output_dir / "SDWI_(Raw).png")

        print(f"Saved enlarged images for month: {month}")
