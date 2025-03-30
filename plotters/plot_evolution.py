import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_evolution():

    # Lists to store data for plotting
    month_indices = []  # X-axis (numeric month indices)
    month_labels = []  # X-axis labels (month names)
    sdwi_means = []  # Y-axis (mean SDWI value per month)

    # Process each month
    for idx, month in enumerate(MONTHS):

        # Load VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, True)

        # Compute SDWI index
        sdwi = get_false_color_image(vv_img, vh_img)[:, :, 2]

        # Normalize SDWI to [0, 1]
        sdwi_norm = (sdwi - np.min(sdwi)) / (np.max(sdwi) - np.min(sdwi) + 1e-10)

        # Store data for this month
        month_indices.append(idx + 1)  # 1-based month index
        month_labels.append(month.capitalize())
        sdwi_means.append(np.mean(sdwi_norm))

    # Create smooth interpolated curve
    x_smooth = np.linspace(min(month_indices), max(month_indices), 300)

    # Create a smooth interpolation
    pchip_interp = PchipInterpolator(month_indices, sdwi_means)
    y_smooth = pchip_interp(x_smooth)

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name="Trend line",
            marker=dict(color="black", size=12, symbol="circle"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Monthly Evolution of High SDWI Values",
        xaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=month_indices,
            ticktext=month_labels,
        ),
        yaxis=dict(title="SDWI Value", range=[0.2, 0.6]),
        height=600,
        width=1200,
        legend_title="Month",
        hovermode="closest",
    )

    # Show the figure
    fig.show()

    # Optionally save the figure
    fig.write_html("results/monthly_sdwi_evolution.html")
