# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy
from scipy import ndimage

# Graphs and plots imports
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

import pathlib
from pathlib import Path

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

mpl_config = Path(pathlib.Path(__file__).parent.resolve() / 'custom_matplotlibrc')
plt.style.use(mpl_config)
plt.rcParams["figure.figsize"] = (9,6)


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

# RGB overlay colours (matching the previous matplotlib implementation)
AXON_RGB = np.array([0, 0, 255], dtype=np.float32)     # blue
MYELIN_RGB = np.array([255, 0, 0], dtype=np.float32)   # red
OVERLAY_ALPHA = 0.5


def _to_uint8(arr):
    """Scale an arbitrary-dtype array to uint8 (matches imshow's autoscaling)."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (((a - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _to_rgb(image):
    """Normalise a 2D grayscale or 3D RGB(A) array to uint8 HxWx3."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        g = _to_uint8(arr)
        return np.repeat(g[:, :, None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return _to_uint8(arr)
    raise ValueError("unexpected image shape {}".format(arr.shape))


def _expand_slice(sl, buffer_pixels, shape):
    """Grow a (yslice, xslice) bounding box by buffer_pixels, clipped to image."""
    ys, xs = sl
    return (
        slice(max(0, ys.start - buffer_pixels), min(shape[0], ys.stop + buffer_pixels)),
        slice(max(0, xs.start - buffer_pixels), min(shape[1], xs.stop + buffer_pixels)),
    )


def _blend(rgb_crop, mask, colour, alpha=OVERLAY_ALPHA):
    """Alpha-composite a flat colour where mask is True. Modifies rgb_crop in place.

    Equivalent to matplotlib's imshow(rgba_overlay) at alpha=0.5.
    """
    if mask.any():
        rgb_crop[mask] = rgb_crop[mask] * (1.0 - alpha) + colour * alpha


class MetricsQA:
    def __init__(self, morphometrics_file):
        """
        :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
        :return: the pixel size value.
        """

        self.file_name = Path(morphometrics_file)

        self.df = pd.read_csv(morphometrics_file)

    def list_metrics(self):
        print('\n'.join(list(self.df.columns.values[3:])))

    def plot(self, metric_name, save_folder = None, quiet = False):
        x = self.df[metric_name].to_numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])

        # bins='fd' uses the Freedman Diaconis Estimator to find the optimal number of bins
        count, bins, ignored = ax1.hist(x,bins='fd', histtype='bar', ec='black')
        ax1.set(xlabel=metric_name, ylabel='Count') 

        ax2.axis('off')
        ax2.set(title='Stats')

        ax2.annotate('mean: ', xy=(0, 0.9))
        ax2.annotate('median: ', xy=(0, 0.8))
        ax2.annotate('std: ', xy=(0, 0.7))
        ax2.annotate('iqr: ', xy=(0, 0.6))
        ax2.annotate('min: ', xy=(0, 0.5))
        ax2.annotate('max: ', xy=(0, 0.4))
        ax2.annotate('NaNs #:', xy=(0, 0.3))

        ax2.annotate(
            np.format_float_positional(np.nanmean(x), precision=2, trim='0'),
            xy=(1, 0.9)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmedian(x), precision=2, trim='0'),
            xy=(1, 0.8)
            )
        ax2.annotate(
            np.format_float_positional(np.nanstd(x), precision=2, trim='0'),
            xy=(1, 0.7)
            )
        ax2.annotate(
            np.format_float_positional(scipy.stats.iqr(x[~np.isnan(x)]), precision=2, trim='0'),
            xy=(1, 0.6))
        
        ax2.annotate(
            np.format_float_positional(np.nanmin(x), precision=2, trim='0'),
            xy=(1, 0.5)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmax(x), precision=2, trim='0'),
            xy=(1, 0.4)
            )
        ax2.annotate(np.sum(np.isnan(x)), xy=(1, 0.3))
        if quiet == False:
            fig.show()

        if save_folder is not None:
            plt.savefig(Path(Path(save_folder) / metric_name))

        # Release the figure when running headless/batch, otherwise matplotlib
        # accumulates every figure ever created (warns past 20, then leaks).
        if quiet:
            plt.close(fig)

        mean = np.format_float_positional(np.nanmean(x), precision=2, trim='0')
        std = np.format_float_positional(np.nanstd(x), precision=2, trim='0')

        return (mean, std)
    
    def plot_all(self, save_folder=None, quiet=False):
        metric_list = list(self.df.columns.values[3:])

        for metric in metric_list:
            if self.df[metric].to_numpy().dtype==np.float64:
                self.plot(metric, save_folder, quiet)

    # -----------------------------------------------------------------------
    # internal
    # -----------------------------------------------------------------------

    def _precompute_ranks(self):
        """Vectorized ranks/percentiles for every axon, computed once.

        Previously each of these six ranks was recomputed over the full column
        inside the per-axon loop, i.e. 6*n full sorts to obtain 6*n numbers.
        """
        df = self.df
        out = {'n': len(df)}
        for key, col in (
            ('diameter', 'axon_diam (um)'),
            ('thickness', 'myelin_thickness (um)'),
            ('gratio', 'gratio'),
        ):
            filled = df[col].fillna(-1)
            out[key] = {
                'pct': filled.rank(pct=True).to_numpy() * 100.0,
                'rank': filled.rank(method='min').to_numpy(),
            }
        return out

    # -----------------------------------------------------------------------
    # flagging
    # -----------------------------------------------------------------------

    def get_flagged_objects(
        self,
        im_axonmyelin_label,
        save_folder,
        gratio_max=0.99,
        area_mode='min_multiple',
        area_factor=5,
        area_percentile=1.0,
    ):
        """Flag suspicious objects and write a mask of them.

        Default behaviour is unchanged from the original implementation.

        :param area_mode: 'min_multiple' (default, original behaviour) flags
            objects at or below area_factor * the smallest object in the image.
            Note this keys off a single object, so one segmentation artifact
            rescales the threshold for the whole image. 'percentile' flags the
            bottom area_percentile% instead, which is stable across images.
        :param save_folder: where to write flagged_objects.png. Pass None to
            skip writing (mask is still returned).
        """
        df = self.df
        axonmyelin_img = np.asarray(im_axonmyelin_label)

        flagged_objects = np.array([])
        flagged_objects = np.append(
            flagged_objects, df.loc[df['gratio'] >= gratio_max].index.to_numpy()
        )

        for col in ('axon_area (um^2)', 'myelin_area (um^2)'):
            if col not in df:
                continue
            if area_mode == 'percentile':
                vals = df[col].to_numpy(dtype=float)
                if not np.isfinite(vals).any():
                    continue
                thresh = np.nanpercentile(vals, area_percentile)
            else:
                thresh = min(df[col]) * area_factor
            flagged_objects = np.append(
                flagged_objects, df.loc[df[col] <= thresh].index.to_numpy()
            )

        flagged_objects = np.unique(flagged_objects)

        # np.isin() paints every flagged object in a single pass. The original
        # ran a full-image np.where() per flagged object.
        mask = np.isin(
            np.rint(axonmyelin_img).astype(np.int64),
            (flagged_objects + 1).astype(np.int64),
        ).astype(axonmyelin_img.dtype)

        if save_folder is not None:
            ads.imwrite(Path(save_folder) / 'flagged_objects.png', mask * 255)

        return (flagged_objects, mask)

    # -----------------------------------------------------------------------
    # closeups
    # -----------------------------------------------------------------------

    def generate_axon_closeups(
        self,
        qa_folder,
        image,
        axon_label,
        myelin_label,
        im_axonmyelin_label,
        buffer_pixels=20,
        only_ids=None,
        min_crop_px=None,
        max_crop_px=None,
    ):
        """Generate closeup images of each axon with overlay using real image data.

        :param only_ids: iterable of 0-based axon ids to render. None (default)
            renders every axon, as before. Pass the flagged ids from
            get_flagged_objects() to render only the axons a human will look at
            -- this is the difference between ~25s and ~1s on a 1000-axon image.
        :param min_crop_px: upscale (nearest-neighbour) any crop whose long edge
            is below this. The previous implementation resampled every crop up
            to ~1200px as a side effect of figsize*dpi; native resolution is
            sharper and far smaller, but set this if the report's CSS expects
            large images.
        :param max_crop_px: downscale (Lanczos) any crop whose long edge exceeds
            this. Useful for very large axons in high-resolution images.
        """
        qa_folder = Path(qa_folder)
        qa_folder.mkdir(parents=True, exist_ok=True)

        base_rgb = _to_rgb(image).astype(np.float32)
        axon_bool = np.asarray(axon_label).astype(bool)
        myelin_bool = np.asarray(myelin_label).astype(bool)

        labels = np.asarray(im_axonmyelin_label)
        labels = (np.rint(labels) if labels.dtype.kind == 'f' else labels).astype(np.int32)
        shape = labels.shape

        # One pass over the label image yields every bounding box, indexed by
        # (label - 1). The original scanned the entire image once per axon.
        objects = ndimage.find_objects(labels)

        ranks = self._precompute_ranks()
        n_axons = ranks['n']

        target_ids = range(n_axons) if only_ids is None else [int(i) for i in only_ids]

        axon_data = []
        for axon_id in target_ids:
            if axon_id < 0 or axon_id >= n_axons:
                continue

            # labels start at 1
            current_axon_id = axon_id + 1
            if current_axon_id > len(objects) or objects[current_axon_id - 1] is None:
                print(f"Warning: No pixels found for axon {axon_id} (ID: {current_axon_id})")
                continue

            sl = _expand_slice(objects[current_axon_id - 1], buffer_pixels, shape)

            image_crop = base_rgb[sl].copy()
            region = labels[sl] == current_axon_id
            axon_current_mask = axon_bool[sl] & region
            myelin_current_mask = myelin_bool[sl] & region

            original_path = qa_folder / f'axon_{axon_id}_original.png'
            labeled_path = qa_folder / f'axon_{axon_id}_labeled.png'

            orig_img = Image.fromarray(image_crop.astype(np.uint8))

            _blend(image_crop, axon_current_mask, AXON_RGB)
            _blend(image_crop, myelin_current_mask, MYELIN_RGB)
            lab_img = Image.fromarray(image_crop.astype(np.uint8))

            if max_crop_px and max(orig_img.size) > max_crop_px:
                s = max_crop_px / max(orig_img.size)
                size = (max(1, int(orig_img.width * s)), max(1, int(orig_img.height * s)))
                orig_img = orig_img.resize(size, Image.LANCZOS)
                lab_img = lab_img.resize(size, Image.LANCZOS)
            elif min_crop_px and max(orig_img.size) < min_crop_px:
                s = min_crop_px / max(orig_img.size)
                size = (max(1, int(orig_img.width * s)), max(1, int(orig_img.height * s)))
                orig_img = orig_img.resize(size, Image.NEAREST)
                lab_img = lab_img.resize(size, Image.NEAREST)

            orig_img.save(original_path, optimize=True)
            lab_img.save(labeled_path, optimize=True)

            row = self.df.iloc[axon_id]
            axon_data.append({
                'id': axon_id,
                'diameter': float(row['axon_diam (um)']),
                'thickness': float(row['myelin_thickness (um)']),
                'gratio': float(row['gratio']),
                'diameterPercentile': f"{ranks['diameter']['pct'][axon_id]:.1f}",
                'thicknessPercentile': f"{ranks['thickness']['pct'][axon_id]:.1f}",
                'gratioPercentile': f"{ranks['gratio']['pct'][axon_id]:.1f}",
                'diameterRank': f"{int(ranks['diameter']['rank'][axon_id])} of {n_axons}",
                'thicknessRank': f"{int(ranks['thickness']['rank'][axon_id])} of {n_axons}",
                'gratioRank': f"{int(ranks['gratio']['rank'][axon_id])} of {n_axons}",
                'imagePath': str(original_path.name),
                'labeledImagePath': str(labeled_path.name)
            })
        
        return axon_data

    def save_seg_overlay(self, image, axon_label, myelin_label, qa_folder):
        """Save overlay of axons and myelin.

        Fixes a bug in the previous implementation: it built an RGB *zeros*
        array and drew it over the whole figure at alpha=0.5, so the black
        background darkened the entire base image. Only labelled pixels are
        touched here.
        """
        qa_folder = Path(qa_folder)
        qa_folder.mkdir(parents=True, exist_ok=True)

        base_rgb = _to_rgb(image)
        Image.fromarray(base_rgb).save(qa_folder / 'base_image.png', optimize=True)

        overlaid = base_rgb.astype(np.float32)
        _blend(overlaid, np.asarray(axon_label).astype(bool), AXON_RGB)
        _blend(overlaid, np.asarray(myelin_label).astype(bool), MYELIN_RGB)
        Image.fromarray(overlaid.astype(np.uint8)).save(
            qa_folder / 'segmentation_overlay.png', optimize=True
        )
