import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_log_comparison():
    # Define parameter values
    a_values = [1, 5, 10, 15, 20]
    b_values = [1, 2, 4, 6, 8]

    # Create subplot titles
    subplot_titles = []
    for a in a_values:
        for b in b_values:
            subplot_titles.append(f'ln({a}x)^{b}')

    # Create a grid of subplots
    fig = make_subplots(
        rows=len(a_values), 
        cols=len(b_values),
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Plot each combination of ln(ax)^b
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            # Define x range from a small positive number to a*65025
            x_max = a * 65025
            x = np.linspace(0.01, x_max, 1000)  # Avoid ln(0)
            
            # Calculate ln(ax)^b
            y = np.power(np.log(a * x), b)
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    showlegend=False,
                    line=dict(color='#1f77b4')
                ),
                row=i+1, col=j+1
            )

    # Update layout
    fig.update_layout(
        title_text='Comparison of ln(ax)^b functions',
        width=1500,
        height=1200,
        template='plotly_white'
    )

    # Add row and column labels
    for i, a in enumerate(a_values):
        fig.update_yaxes(title_text=f'a={a}', row=i+1, col=1)
    
    for j, b in enumerate(b_values):
        fig.update_xaxes(title_text=f'b={b}', row=len(a_values), col=j+1)

    # Add grid to all subplots
    for i in range(1, len(a_values)+1):
        for j in range(1, len(b_values)+1):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=j)

    # Show plot
    fig.show()

    # Save plot
    fig.write_html('log_comparison.html')