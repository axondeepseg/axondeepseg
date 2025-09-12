import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
print("Hello world")
# --- Example Data ---
df1 = pd.DataFrame({"axon_diameter": [1, 2, 3, 4], "count": [10, 30, 50, 20]})
df2 = pd.DataFrame({"myelin_thickness": [0.2, 0.3, 0.4, 0.5], "count": [15, 40, 35, 10]})

# Plotly figures
fig1_html = px.bar(df1, x="axon_diameter", y="count",
                   title="Axon Diameter Distribution").to_html(full_html=False, include_plotlyjs=False)
fig2_html = px.bar(df2, x="myelin_thickness", y="count",
                   title="Myelin Thickness Distribution").to_html(full_html=False, include_plotlyjs=False)

# Example image thumbnails (replace with your real histology images)
# For demo, these can be small PNGs in the same folder.
example_images = ["example1.png", "example2.png"]

# --- Build Sections Dynamically ---
sections = {
    "Morphometrics": [
        {"type": "plot", "html": fig1_html},
        {"type": "plot", "html": fig2_html},
    ],
    "Segmentation QC": [
        {"type": "image", "src": img} for img in example_images
    ],
    "Summary": [
        {"type": "plot", "html": fig1_html},  # just reusing for demo
    ],
}

# --- Render Jinja2 template ---
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("report_template.html")

html_out = template.render(sections=sections)

with open("AxonDeepSeg_QA_demo.html", "w") as f:
    f.write(html_out)

print("✅ Generated AxonDeepSeg_QA_demo.html")
