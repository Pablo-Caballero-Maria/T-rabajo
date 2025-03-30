import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cosine

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_heatmap_py():
    # Constants
    WATER_THRESHOLD = 0.8  # Threshold for water classification
    vv_img, vh_img = load_bands_from_month("jan", True) # get shape of image from first month (all images have same shape)
    PIXEL_SIZE_KM = 31 / np.sqrt(vv_img.shape[0] * vv_img.shape[1])  # Size of one pixel in km²

    # Data collection
    water_areas = []  # Water area in km² for each month
    month_names = []
    water_masks = []  # Binary masks of water pixels

    for month in MONTHS:
        month_names.append(month.capitalize())

        # Load VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, True)

        # Compute SDWI index
        sdwi = get_false_color_image(vv_img, vh_img)[:, :, 2]

        # Normalize SDWI to [0, 1]
        sdwi_norm = (sdwi - np.min(sdwi)) / (np.max(sdwi) - np.min(sdwi) + 1e-10)

        # Apply threshold to identify water pixels (1 = water, 0 = not water)
        water_mask = (sdwi_norm >= WATER_THRESHOLD).astype(int)
        water_masks.append(water_mask)
        
        # Calculate water area in km²
        water_pixel_count = np.sum(water_mask)
        water_area_km2 = water_pixel_count * (PIXEL_SIZE_KM ** 2)
        water_areas.append(water_area_km2)
        
        print(f"{month.capitalize()}: {water_area_km2:.2f} km² of water ({water_pixel_count} pixels)")

    # Create similarity matrix based on water surface area
    similarity_matrix = np.zeros((len(MONTHS), len(MONTHS)))

    for i in range(len(MONTHS)):
        for j in range(len(MONTHS)):
            # Calculate similarity as 1 minus the normalized absolute difference
            # This gives 1 for identical areas and approaches 0 for very different areas
            max_area = max(water_areas[i], water_areas[j])
            if max_area > 0:
                similarity_matrix[i, j] = 1 - abs(water_areas[i] - water_areas[j]) / max_area
            else:
                similarity_matrix[i, j] = 1  # Both have zero water area = perfect similarity

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=month_names,
            y=month_names,
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(
                title="Similarity",
            ),
            text=[[f"{similarity_matrix[i][j]:.2f}<br>Water area: {water_areas[i]:.2f} km² vs {water_areas[j]:.2f} km²" 
                   for j in range(len(MONTHS))] 
                   for i in range(len(MONTHS))],
            hoverinfo="text+z"
        )
    )

    # Update layout
    fig.update_layout(
        title="Similarity Between Monthly Water Surface Area (SDWI ≥ 0.8)",
        xaxis_title="Month",
        yaxis_title="Month",
        height=800,
        width=800,
    )

    # Show the figure
    fig.show()

    # Optionally save the figure
    fig.write_html("results/water_area_similarity_heatmap.html")