import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cosine

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_heatmap_py():

    # Data collection
    month_data = []
    month_names = []

    for month in MONTHS:
        month_names.append(month.capitalize())

        # Load VV and VH bands
        vv_img, vh_img = load_bands_from_month(month, True)

        # Compute SDWI index
        sdwi = get_false_color_image(vv_img, vh_img)[:, :, 2]

        # Normalize SDWI to [0, 1]
        sdwi_norm = (sdwi - np.min(sdwi)) / (np.max(sdwi) - np.min(sdwi) + 1e-10)

        # Flatten and append to our dataset
        month_data.append(sdwi_norm.flatten())

    # Create bins for the histogram
    num_bins = 50
    bin_ranges = np.linspace(0, 1, num_bins + 1)

    # Compute histogram for each month
    histograms = []

    for data in month_data:
        hist, _ = np.histogram(data, bins=bin_ranges, density=True)
        histograms.append(hist)

    # Create similarity matrix (using cosine similarity)
    similarity_matrix = np.zeros((len(MONTHS), len(MONTHS)))

    for i in range(len(MONTHS)):
        for j in range(len(MONTHS)):
            # Cosine similarity = 1 - cosine distance
            # Higher value means more similar histograms
            similarity_matrix[i, j] = 1 - cosine(histograms[i], histograms[j])

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
        )
    )

    # Update layout
    fig.update_layout(
        title="Similarity Between Monthly SDWI Distributions",
        xaxis_title="Month",
        yaxis_title="Month",
        height=800,
        width=800,
    )

    # Show the figure
    fig.show()

    # Optionally save the figure
    fig.write_html("results/sdwi_similarity_heatmap.html")
