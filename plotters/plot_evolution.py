import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_evolution():
    # Constants
    WATER_THRESHOLD = 0.8  # Threshold for water classification
    
    # Get image dimensions from first month
    vv_img, vh_img = load_bands_from_month(MONTHS[0], True)
    PIXEL_SIZE_KM = 31 / np.sqrt(vv_img.shape[0] * vv_img.shape[1])  # Size of one pixel in km²

    # Lists to store data for plotting
    month_indices = []  # X-axis (numeric month indices)
    month_labels = []  # X-axis labels (month names)
    water_areas = []  # Y-axis (water surface area per month in km²)

    # Process each month
    for idx, month in enumerate(MONTHS):
        # Load VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, True)

        # Compute SDWI index
        sdwi = get_false_color_image(vv_img, vh_img)[:, :, 2]

        # Normalize SDWI to [0, 1]
        sdwi_norm = (sdwi - np.min(sdwi)) / (np.max(sdwi) - np.min(sdwi) + 1e-10)

        # Apply threshold to identify water pixels (1 = water, 0 = not water)
        water_mask = (sdwi_norm >= WATER_THRESHOLD).astype(int)
        
        # Calculate water area in km²
        water_pixel_count = np.sum(water_mask)
        water_area_km2 = water_pixel_count * (PIXEL_SIZE_KM ** 2)
        
        # Store data for this month
        month_indices.append(idx + 1)  # 1-based month index
        month_labels.append(month.capitalize())
        water_areas.append(water_area_km2)
        
        print(f"{month.capitalize()}: {water_area_km2:.2f} km² of water ({water_pixel_count} pixels)")

    # Create smooth interpolated curve
    x_smooth = np.linspace(min(month_indices), max(month_indices), 300)

    # Create a smooth interpolation
    pchip_interp = PchipInterpolator(month_indices, water_areas)
    y_smooth = pchip_interp(x_smooth)

    # Create figure
    fig = go.Figure()

    # Add smooth curve
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name="Trend line",
            line=dict(color="blue", width=2),
        )
    )

    # Calculate min and max for y-axis with some padding
    y_min = max(0, min(water_areas) * 0.9)  # Don't go below zero
    y_max = max(water_areas) * 1.1  # Add 10% padding

    # Update layout
    fig.update_layout(
        title="Monthly Evolution of Water Surface Area (SDWI ≥ 0.8)",
        xaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=month_indices,
            ticktext=month_labels,
        ),
        yaxis=dict(
            title="Water Surface Area (km²)",
            range=[y_min, y_max]
        ),
        height=600,
        width=1200,
        legend_title="Data",
        hovermode="closest",
    )

    # Show the figure
    fig.show()

    # Optionally save the figure
    fig.write_html("results/monthly_water_area_evolution.html")