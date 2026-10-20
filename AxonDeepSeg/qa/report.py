"""
Generate the QA report as a single, self-contained HTML file.

Everything the report needs is embedded in the HTML: the image, a label map telling which
axon each pixel belongs to, the overview and the histograms. The close-ups of the Single Axon
Viewer are drawn by the browser from the embedded image and label map, so the report can be
moved or shared as one file.
"""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt

import AxonDeepSeg
from AxonDeepSeg.qa.metrics_qa import MetricsQA

REPORT_NAME = "AxonDeepSeg_QA_Report.html"
OVERVIEW_MAX_SIZE = 2000  # px, longest side of the overview image on the Summary tab
HISTOGRAMS = [
    ("Axon Diameter", "axon_diam (um)"),
    ("Myelin Thickness", "myelin_thickness (um)"),
    ("g-ratio", "gratio"),
    ("Axon Area", "axon_area (um^2)"),
    ("Myelin Area", "myelin_area (um^2)"),
]


def _to_uint8_gray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image[..., :3].astype(np.uint8)).convert("L"))
    if image.dtype != np.uint8:
        lo, hi = np.percentile(image, (0.5, 99.5))
        image = np.clip((image - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return image


def _data_uri(pil_image, fmt="PNG", **save_kwargs):
    buffer = BytesIO()
    pil_image.save(buffer, format=fmt, **save_kwargs)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_label_map(im_axonmyelin_label, axon_mask, myelin_mask):
    """
    Pack the instance label map and the axon/myelin classes into one lossless RGB PNG:
    R and G hold the axon label (label = morphometrics row index + 1, 16 bits),
    B holds the class (0 background, 1 axon, 2 myelin).
    """
    labels = np.asarray(im_axonmyelin_label).astype(np.int64)
    if labels.max() > 65535:
        raise ValueError("The QA report supports up to 65535 axons per image.")
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = labels & 255
    rgb[..., 1] = labels >> 8
    rgb[..., 2][np.asarray(myelin_mask).astype(bool)] = 2
    rgb[..., 2][np.asarray(axon_mask).astype(bool)] = 1
    return _data_uri(Image.fromarray(rgb, mode="RGB"), "PNG", optimize=True)


def build_axon_data(df, im_axonmyelin_label):
    """One entry per row of the morphometrics file, with its bounding box in the image."""
    labels = np.asarray(im_axonmyelin_label)
    boxes = ndimage.find_objects(labels.astype(np.int32))
    n_axons = len(df)

    def ranks(column):
        values = df[column].fillna(-1)
        return values.rank(pct=True) * 100, values.rank(method="min")

    d_pct, d_rank = ranks("axon_diam (um)")
    t_pct, t_rank = ranks("myelin_thickness (um)")
    g_pct, g_rank = ranks("gratio")
    border = df["image_border_touching"] if "image_border_touching" in df.columns else pd.Series(False, index=df.index)

    axon_data = []
    for axon_id in range(n_axons):
        row = df.iloc[axon_id]
        box = boxes[axon_id] if axon_id < len(boxes) else None
        axon_data.append({
            "id": axon_id,
            "diameter": float(row["axon_diam (um)"]),
            "thickness": float(row["myelin_thickness (um)"]),
            "gratio": float(row["gratio"]),
            "border": bool(border.iloc[axon_id]) if pd.notna(border.iloc[axon_id]) else False,
            "diameterPercentile": f"{d_pct.iloc[axon_id]:.1f}",
            "thicknessPercentile": f"{t_pct.iloc[axon_id]:.1f}",
            "gratioPercentile": f"{g_pct.iloc[axon_id]:.1f}",
            "diameterRank": f"{int(d_rank.iloc[axon_id])} of {n_axons}",
            "thicknessRank": f"{int(t_rank.iloc[axon_id])} of {n_axons}",
            "gratioRank": f"{int(g_rank.iloc[axon_id])} of {n_axons}",
            # bounding box in image pixels: [x_min, y_min, x_max, y_max], None if the axon has no pixels
            "bbox": None if box is None else [box[1].start, box[0].start, box[1].stop, box[0].stop],
        })
    return axon_data


def overview_images(image, axon_mask, myelin_mask):
    """Downscaled overview of the image, with and without the segmentation overlay."""
    gray = Image.fromarray(image)
    scale = min(1.0, OVERVIEW_MAX_SIZE / max(gray.size))
    size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
    gray_small = np.asarray(gray.resize(size, Image.LANCZOS)).astype(np.float32)
    axon_small = np.asarray(Image.fromarray(np.asarray(axon_mask).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    myelin_small = np.asarray(Image.fromarray(np.asarray(myelin_mask).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    rgb = np.stack([gray_small] * 3, axis=-1)
    for mask, color in ((myelin_small, (0, 0, 255)), (axon_small, (255, 0, 0))):
        rgb[mask] = 0.6 * rgb[mask] + 0.4 * np.array(color)
    original = _data_uri(Image.fromarray(gray_small.astype(np.uint8)), "JPEG", quality=85)
    labeled = _data_uri(Image.fromarray(rgb.astype(np.uint8)), "JPEG", quality=85)
    return original, labeled


def generate_qa_report(
    image, axon_mask, myelin_mask, im_axonmyelin_label, morphometrics_file, qa_folder,
    image_name=None, default_filters=None, image_quality=85,
):
    """
    Generate the QA report for one image.

    Parameters
    ----------
    image : numpy.ndarray
        The microscopy image (grayscale or RGB).
    axon_mask, myelin_mask : numpy.ndarray
        Binary axon and myelin masks used for the morphometrics.
    im_axonmyelin_label : numpy.ndarray
        Instance label map returned by get_axon_morphometrics (label = row index + 1).
    morphometrics_file : str or pathlib.Path
        Morphometrics CSV computed from these masks.
    qa_folder : str or pathlib.Path
        Folder where the report (and the histogram PNGs) are written.
    image_name : str, optional
        Name used in the report and in the exported flags file name.
    default_filters : dict, optional
        Initial values of the axon filters of the viewer, e.g.
        {"min_diameter": 0.3, "max_diameter": 3.0, "max_gratio": 0.95, "exclude_border": True}.
        The user can change them in the report.
    image_quality : int
        JPEG quality of the embedded image.

    Returns
    -------
    pathlib.Path
        Path of the HTML report.
    """
    qa_folder = Path(qa_folder)
    qa_folder.mkdir(parents=True, exist_ok=True)
    image = _to_uint8_gray(image)
    axon_mask = np.asarray(axon_mask).astype(bool)
    myelin_mask = np.asarray(myelin_mask).astype(bool) & ~axon_mask

    qa = MetricsQA(morphometrics_file)
    stats = {column: qa.plot(column, save_folder=qa_folder, quiet=True) for _, column in HISTOGRAMS}
    plt.close("all")
    histograms = []
    for name, column in HISTOGRAMS:
        png = qa_folder / f"{column}.png"
        histograms.append({"name": name, "src": _data_uri(Image.open(png), "PNG")})

    original_src, labeled_src = overview_images(image, axon_mask, myelin_mask)

    sections = {
        "Summary": [
            {"type": "stats", "stats": [
                {"label": "Axon Diameter (<span class=\"unit\">µm</span>)", "value": "{} ± {}".format(*stats["axon_diam (um)"])},
                {"label": "Myelin Thickness (<span class=\"unit\">µm</span>)", "value": "{} ± {}".format(*stats["myelin_thickness (um)"])},
                {"label": "g-ratio", "value": "{} ± {}".format(*stats["gratio"])},
            ]},
            {"type": "text", "content": "<h3>Quality Assessment</h3><p>This report shows automated quality control metrics for axon segmentation.</p>"},
            {"type": "segmented", "labeled_src": labeled_src, "original_src": original_src},
        ],
        "Histograms": [{"type": "histogram_viewer", "histograms": histograms}],
    }

    env = Environment(loader=FileSystemLoader(str((Path(AxonDeepSeg.__file__).parent / "qa").resolve())))
    html = env.get_template("report_template.html").render(
        sections=sections,
        axon_data=build_axon_data(qa.df, im_axonmyelin_label),
        image_src=_data_uri(Image.fromarray(image), "JPEG", quality=image_quality),
        label_src=encode_label_map(im_axonmyelin_label, axon_mask, myelin_mask),
        image_size=[int(image.shape[1]), int(image.shape[0])],
        default_filters=default_filters or {},
        sample_id=Path(morphometrics_file).stem,
        image_name=image_name or Path(morphometrics_file).stem,
        report_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        software_version=AxonDeepSeg.__version__,
    )
    report_path = qa_folder / REPORT_NAME
    report_path.write_text(html, encoding="utf-8")
    return report_path
