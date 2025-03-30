import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_sdwi_segmentated():
    # Create a 4x3 subplot grid
    fig = make_subplots(
        rows=3, cols=4, subplot_titles=[month.capitalize() for month in MONTHS]
    )

    threshold = 0.8

    # Process each month
    for idx, month in enumerate(MONTHS):

        row = idx // 4 + 1
        col = idx % 4 + 1

        # Load VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, True)

        # Compute SDWI index
        sdwi_norm = get_false_color_image(vv_img, vh_img)[
            :, :, 2
        ]  # <- this is in [0,255]
        # Normalize SDWI to [0, 1]
        sdwi_norm = (sdwi_norm - np.min(sdwi_norm)) / (
            np.max(sdwi_norm) - np.min(sdwi_norm) + 1e-10
        )

        # Apply thresholding for segmentation
        segmented = np.zeros_like(sdwi_norm)
        segmented[sdwi_norm <= threshold] = 0.0  # Black
        segmented[sdwi_norm > threshold] = 1.0  # White

        # Convert single-channel image to 3-channel grayscale
        segmented_rgb = np.stack([segmented] * 3, axis=-1)

        # Scale from [0,1] to [0,255] for plotly to display correctly
        segmented_rgb = (segmented_rgb * 255).astype(np.uint8)

        # Add segmented SDWI image to the plotly subplot
        fig.add_trace(go.Image(z=segmented_rgb), row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="Monthly Sentinel-1 Radar Images (Segmented SDWI Index)",
        height=900,
        width=1200,
    )

    # Remove axes
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Show the figure
    fig.show()

    # Optionally save the figure
    fig.write_html("results/monthly_segmented_sdwi.html")
