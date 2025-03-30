import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.img_utils import MONTHS, load_bands_from_month


def plot_histograms(band):
    # Create a 4x3 subplot grid
    fig = make_subplots(
        rows=3, cols=4, subplot_titles=[month.capitalize() for month in MONTHS]
    )

    # Process each month
    for idx, month in enumerate(MONTHS):
        row = idx // 4 + 1
        col = idx % 4 + 1

        vv_img, vh_img = load_bands_from_month(month, True)

        # Select the appropriate band
        img = vv_img if band == "vv" else vh_img

        # Create histogram for the band data
        fig.add_trace(
            go.Histogram(
                x=img.flatten(),
                nbinsx=50,
                name=month.capitalize(),
                xbins=dict(start=0, end=255, size=5),
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title_text=f"Monthly Sentinel-1 Radar {band.upper()} Band Histograms",
        height=900,
        width=1200,
    )

    # Set x-axis range for all subplots to be 0-255
    for i in range(1, 4):  # rows
        for j in range(1, 5):  # columns
            fig.update_xaxes(range=[0, 255], row=i, col=j)

    # Show the figure
    fig.show()

    # Save the figure
    fig.write_html(f"results/monthly_{band}_histograms.html")
