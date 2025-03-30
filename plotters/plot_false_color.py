import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.img_utils import (MONTHS, get_false_color_image,
                             load_bands_from_month)


def plot_false_color():
    # Create a 4x3 subplot grid
    fig = make_subplots(
        rows=3, cols=4, subplot_titles=[month.capitalize() for month in MONTHS]
    )

    # Process each month
    for idx, month in enumerate(MONTHS):

        row = idx // 4 + 1
        col = idx % 4 + 1

        vv_img, vh_img = load_bands_from_month(month, True)
        false_color = get_false_color_image(vv_img, vh_img)

        # Add image to the plotly subplot
        fig.add_trace(go.Image(z=false_color), row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="Monthly Sentinel-1 Radar Images (False Color)",
        height=900,
        width=1200,
    )

    # Remove axes
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Show the figure
    fig.show()

    # Save the figure
    fig.write_html("results/monthly_radar_images.html")
