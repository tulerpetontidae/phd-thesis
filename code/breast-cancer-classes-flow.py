import plotly.graph_objects as go
import numpy as np

# A dictionary to convert CSS names to RGB
css_to_rgb = {
    'orangered': (255, 69, 0),
    'mediumseagreen': (60, 179, 113),
    'darkorange': (255, 140, 0),
    'dodgerblue': (30, 144, 255),
    'grey': (128, 128, 128)
}

# Convert RGB to RGBA with alpha
def rgb_to_rgba(rgb_tuple, alpha=1):
    return f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{alpha})"

# Define your data
ihc_labels = ['HR+/HER2–', 'HR+/HER2+', 'HR–/HER2+', 'TNBC']
pam50_labels = ['Luminal A', 'Luminal B', 'HER2', 'Basal', 'Normal']
colors = ['dodgerblue', 'mediumseagreen', 'darkorange', 'orangered', 'grey']

# Convert CSS named colors to RGB, then to RGBA
colors_rgba = [rgb_to_rgba(css_to_rgb[color], 0.6) for color in colors]
node_colors = ['black'] * len(ihc_labels) + [rgb_to_rgba(css_to_rgb[color], 1) for color in colors]

ihc_flows = np.array([0.46, 0.107, 0.186, 0.247])
pam50_percentages_by_ihc = np.array([
    [55, 29, 4, 6, 6],
    [31, 22, 42, 0, 5],
    [4, 6, 74, 9, 7],
    [2, 0, 13, 81, 4]
])

# Prepare data for the Sankey diagram
source = []
target = []
value = []
flow_colors = []
node_colors = ['black'] * len(ihc_labels) + colors  # Black for IHC, and use given colors for PAM50

# Calculate pam50 flows based on ihc flows and percentages
pam50_flows = ihc_flows[:, np.newaxis] * (pam50_percentages_by_ihc / 100)

for i, ihc_label in enumerate(ihc_labels):
    for j, pam50_label in enumerate(pam50_labels):
        source.append(i)
        target.append(len(ihc_labels) + j)
        value.append(pam50_flows[i, j])
        flow_colors.append(colors_rgba[j])

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0),
        label=ihc_labels + pam50_labels,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=flow_colors
    )
)])

# Add title and layout options
fig.update_layout(title_text="IHC to PAM50 Sankey Flow Diagram", font_size=10)

# Show the figure
fig.write_image("code/raw_fig/sankey-cancer-subtypes.pdf")
