import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from utils.denoisers import (denoise_image_bm3d, denoise_image_frost,
                             denoise_image_kuan, denoise_image_lee,
                             denoise_image_median,
                             denoise_image_non_local_means,
                             denoise_image_wiener)
from utils.enlargers import (enlarge_image_bicubic, enlarge_image_bilinear,
                             enlarge_image_fft, enlarge_image_gradients,
                             enlarge_image_lanczos,
                             enlarge_image_nearest_neighbor,
                             enlarge_image_nedi, enlarge_image_spline)
from utils.img_utils import *


def plot_comparison():
    month = "jan"
    vv_img, vh_img = load_bands_from_month(month, False)
    sdwi_norm = get_false_color_image(vv_img, vh_img)[:, :, 2]
    # Define denoisers and enlargers to test
    denoisers = {
        "None": lambda x: x,
        "Median": denoise_image_median,
        "Lee": denoise_image_lee,
        "Frost": denoise_image_frost,
        "Kuan": denoise_image_kuan,
        "Wiener": denoise_image_wiener,
        "Non-local Means": denoise_image_non_local_means,
        "BM3D": denoise_image_bm3d,
    }

    enlargers = {
        "None": lambda x: x,
        "Nearest Neighbor": enlarge_image_nearest_neighbor,
        "Bilinear": enlarge_image_bilinear,
        "Bicubic": enlarge_image_bicubic,
        "Lanczos": enlarge_image_lanczos,
        "Spline": enlarge_image_spline,
        "FFT": enlarge_image_fft,
        "Gradients": enlarge_image_gradients,
        "NEDI": enlarge_image_nedi,
    }

    # Create output directory
    output_dir = Path("results/individual_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    '''
    # Keep track of all processed images
    processed_images = {}
    # Process each combination and save individual PNGs
    for denoiser_name, denoiser_func in denoisers.items():
        for enlarger_name, enlarger_func in enlargers.items():
            print(f"Processing: Denoiser: {denoiser_name}, Enlarger: {enlarger_name}")
            
            # Process image
            sdwi_norm_denoised = denoiser_func(sdwi_norm)
            sdwi_norm_enlarged = enlarger_func(sdwi_norm_denoised)
            
            # Store processed image for final plot
            processed_images[(denoiser_name, enlarger_name)] = sdwi_norm_enlarged
            
            # Create a single figure for this combination
            fig = go.Figure(
                go.Image(z=np.stack([sdwi_norm_enlarged] * 3, axis=-1))
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                width=400,
            )
            
            # Remove axes
            fig.update_xaxes(showticklabels=False, showgrid=False)
            fig.update_yaxes(showticklabels=False, showgrid=False)
            
            # Generate filename
            filename = f"{denoiser_name}_{enlarger_name}.png".replace(" ", "_").lower()
            filepath = output_dir / filename
            
            # Save the figure
            fig.write_image(str(filepath))
    '''        
    print("All individual images saved. Creating comparison grid...")

    # Create a grid of subplots for all combinations
    num_denoisers = len(denoisers)
    num_enlargers = len(enlargers)

    # Create grid without subplot titles
    fig = make_subplots(
        rows=num_denoisers,
        cols=num_enlargers,
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
    )

    # Add each processed image to the appropriate subplot
    for i, (denoiser_name, denoiser_func) in enumerate(denoisers.items(), 1):
        for j, (enlarger_name, enlarger_func) in enumerate(enlargers.items(), 1):
            print(
                f"Adding to grid: Denoiser: {denoiser_name}, Enlarger: {enlarger_name}"
            )
            
            # check if the image is stored locally
            filename = f"{denoiser_name}_{enlarger_name}.png".replace(" ", "_").lower()
            filepath = output_dir / filename
            # Load the image
            processed_img = np.array(Image.open(filepath))
            # processed_img = processed_images[(denoiser_name, enlarger_name)]
            # Add image to subplot
            fig.add_trace(go.Image(z=processed_img[:, :, :3]), row=i, col=j)

    # Update layout for the entire figure
    fig.update_layout(
        title_text="January Radar Imagery",
        height=250 * num_denoisers,
        width=250 * num_enlargers,
        showlegend=False,
        margin=dict(t=100, l=150)  # Make space for column and row labels
    )

    # Remove axes from all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Add enlarger names as column headers at the top
    enlarger_names = list(enlargers.keys())
    for j, enlarger_name in enumerate(enlarger_names):
        fig.add_annotation(
            x=j/(num_enlargers-1) if num_enlargers > 1 else 0.5,  # Normalized position
            y=1.02,  # Just above the top of the grid
            text=enlarger_name,
            showarrow=False,
            font=dict(size=12),
            xref="paper",
            yref="paper",
            xanchor="center"
        )

    # Add denoiser names as row labels on the left
    denoiser_names = list(denoisers.keys())
    for i, denoiser_name in enumerate(denoiser_names):
        fig.add_annotation(
            x=-0.01,  # Just to the left of the grid
            y=1 - (i+0.5)/num_denoisers,  # Normalized position
            text=denoiser_name,
            showarrow=False,
            font=dict(size=12),
            xref="paper",
            yref="paper",
            xanchor="right"
        )

    fig.write_image("results/january_comparison_grid.png")

    print("Comparison grid created at results/january_comparison_grid.png")
