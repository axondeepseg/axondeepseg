"""Background worker threads wiring the standalone GUI to the AxonDeepSeg backend.

Kept intentionally simple: everything runs on CPU (no GPU selection exposed),
one thread per action (segment / morphometrics / download), cancellation is
cooperative (checked between images), and progress/log updates are emitted as
Qt signals so the main window can update itself from the GUI thread.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from AxonDeepSeg import ads_utils as ads
from AxonDeepSeg import segment, postprocessing
from AxonDeepSeg.params import (
    valid_extensions, generated_file_suffixes,
    axon_suffix, myelin_suffix, axonmyelin_suffix,
    unmyelinated_suffix, nerve_suffix,
    axonmyelin_index_suffix, unmyelinated_index_suffix, nerve_index_suffix,
    instance_im_suffix, instance_suffix, diameter_overlay_suffix,
)
from AxonDeepSeg.morphometrics.compute_morphometrics import (
    get_axon_morphometrics, save_axon_morphometrics,
    save_nerve_morphometrics_to_json, remove_outside_nerve, compute_axon_density,
)
from AxonDeepSeg.download_model import download_model


def resolve_segmented_image_path(img_path: Path) -> Path:
    """Follow AxonDeepSeg/segment.py:prepare_inputs's file rename.

    Segmentation converts the input to a '<stem>_grayscale.png' copy when it doesn't
    already match the model's expected channel count/format, and writes its output
    masks against that copy's name instead of the original file's. Downstream code
    (morphometrics) needs to resolve to whichever one the masks actually live next to.
    """
    grayscale_candidate = img_path.parent / (img_path.stem + "_grayscale.png")
    return grayscale_candidate if grayscale_candidate.exists() else img_path


def find_result_masks(img_path: Path) -> dict:
    """Locate whichever segmentation masks exist next to img_path.

    Models can output myelinated axons (axon + myelin), unmyelinated axons
    (uaxon), or both at once (e.g. the Stanford model) — so these are checked
    independently rather than as an either/or.
    """
    def _existing(suffix):
        p = img_path.parent / (img_path.stem + str(suffix))
        return p if p.exists() else None

    return {
        "axon": _existing(axon_suffix),
        "myelin": _existing(myelin_suffix),
        "uaxon": _existing(unmyelinated_suffix),
    }


# Matches the napari plugin's mask colors (AxonDeepSeg/ads_napari/_widget.py) for
# axon/myelin; unmyelinated axons get green since they're a distinct class.
_AXON_COLOR = np.array([0, 0, 255], dtype=np.float32)
_MYELIN_COLOR = np.array([255, 0, 0], dtype=np.float32)
_UAXON_COLOR = np.array([0, 255, 0], dtype=np.float32)


def compose_overlay(image: np.ndarray, axon: Optional[np.ndarray] = None,
                     myelin: Optional[np.ndarray] = None, uaxon: Optional[np.ndarray] = None,
                     alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend segmentation masks over a grayscale image for a quick visual check.

    Returns an (H, W, 3) uint8 RGB array. Axon is blue, myelin is red,
    unmyelinated axon is green.
    """
    if image.ndim == 3:
        image = image[..., 0]
    base = np.stack([image.astype(np.float32)] * 3, axis=-1)

    out = base.copy()
    for mask, color in ((myelin, _MYELIN_COLOR), (axon, _AXON_COLOR), (uaxon, _UAXON_COLOR)):
        if mask is None:
            continue
        thresh = mask.max() / 2 if mask.max() > 0 else 127
        m = mask > thresh
        out[m] = out[m] * (1 - alpha) + color * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def expand_to_image_files(paths: List[str]) -> List[Path]:
    """Expand a mix of file/folder path strings into a flat, deduplicated list of image files."""
    files = []
    seen = set()
    for p in paths:
        path = Path(p)
        candidates = []
        if path.is_dir():
            candidates = sorted(path.iterdir())
        elif path.exists():
            candidates = [path]
        for f in candidates:
            if (
                f.suffix.lower() in valid_extensions
                and not str(f).endswith(generated_file_suffixes)
                and not str(f).endswith("_grayscale.png")
                and f not in seen
            ):
                files.append(f)
                seen.add(f)
    return files


class SegmentThread(QThread):
    """Runs segmentation on a batch of images, optionally chaining morphometrics."""

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.paths: List[str] = []
        self.path_model: Optional[Path] = None
        self.allow_large_images = False
        self.run_morph_after = False
        self.cancel_requested = False
        # Populated on success with one entry per segmented image that has at least
        # one output mask: {"image": Path, "axon": Path|None, "myelin": Path|None}.
        self.segmented_results: List[dict] = []

    @staticmethod
    def _pick_gpu() -> int:
        """Auto-detect a usable GPU. No UI toggle: use GPU 0 if available, else CPU."""
        try:
            n_gpus = ads.check_available_gpus(None)
        except Exception:
            return -1
        return 0 if n_gpus > 0 else -1

    def run(self):
        import nnunetv2.inference.predict_from_raw_data as nnunet_predictor
        from loguru import logger

        _original_tqdm = nnunet_predictor.tqdm

        # segment_images is decorated with @logger.catch(reraise=False), which silently
        # swallows all exceptions and returns None. To surface those errors in the GUI we
        # add a temporary loguru sink that forwards ERROR-level records to the log signal.
        _caught_error: List[str] = []

        def _error_sink(msg):
            _caught_error.append(msg.record["message"])
            self.log.emit(f"Segmentation error: {msg.record['message']}")

        _sink_id = logger.add(_error_sink, level="ERROR", format="{message}")

        def _gui_tqdm(iterable=None, *args, **kwargs):
            kwargs.pop("disable", None)
            if iterable is not None:
                items = list(iterable)
                total = len(items)
                self.progress.emit(0, total, "Segmenting")
                for i, item in enumerate(items):
                    if self.cancel_requested:
                        raise InterruptedError("Segmentation cancelled by user")
                    yield item
                    self.progress.emit(i + 1, total, "Segmenting")
            else:
                yield from _original_tqdm(*args, **kwargs)

        nnunet_predictor.tqdm = _gui_tqdm

        success = False
        try:
            image_files = expand_to_image_files(self.paths)
            if not image_files:
                self.log.emit("No images found in the selected input(s).")
                return
            gpu_id = self._pick_gpu()
            device_desc = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
            self.log.emit(f"Segmenting {len(image_files)} image(s) with {self.path_model.name} ({device_desc})...")
            segment.segment_images(
                path_images=image_files,
                path_model=self.path_model,
                gpu_id=gpu_id,
                verbosity_level=0,
                allow_large_images=self.allow_large_images,
            )
            if _caught_error:
                # @logger.catch swallowed an exception; error already forwarded to the log.
                return
            # Belt and suspenders: some nnU-Net internals swallow the InterruptedError
            # raised from _gui_tqdm and return normally, so re-check the flag here
            # instead of only relying on the exception below.
            if self.cancel_requested:
                self.log.emit("Segmentation cancelled.")
                return
            success = True
            self.log.emit("Segmentation done.")
            self.segmented_results = self._collect_results(image_files)

            if self.run_morph_after:
                self._run_morphometrics_after(image_files)
        except InterruptedError:
            self.log.emit("Segmentation cancelled.")
        except Exception as e:
            self.log.emit(f"Segmentation failed: {e}")
        finally:
            nnunet_predictor.tqdm = _original_tqdm
            logger.remove(_sink_id)
        self.finished_ok.emit(success)

    def _collect_results(self, image_files: List[Path]) -> List[dict]:
        results = []
        for img_path in image_files:
            resolved = resolve_segmented_image_path(img_path)
            masks = find_result_masks(resolved)
            if masks["axon"] is not None or masks["uaxon"] is not None:
                results.append({"image": resolved, **masks})
        return results

    def _run_morphometrics_after(self, image_files: List[Path]):
        from AxonDeepSeg.morphometrics.launch_morphometrics_computation import (
            launch_morphometrics_computation,
        )
        for img_path in image_files:
            resolved = resolve_segmented_image_path(img_path)
            mask_path = resolved.parent / (resolved.stem + str(axonmyelin_suffix))
            if not mask_path.exists():
                continue
            try:
                launch_morphometrics_computation(resolved, mask_path, axon_shape="circle")
                self.log.emit(f"Morphometrics computed for {img_path.name}.")
            except Exception as e:
                self.log.emit(f"Morphometrics failed for {img_path.name}: {e}")


class MorphometricsThread(QThread):
    """Computes morphometrics for a batch of images that already have a segmentation mask.

    Mirrors AxonDeepSeg/morphometrics/launch_morphometrics_computation.py's CLI loop
    (myelinated / unmyelinated / nerve modes), minus the argparse/sys.exit plumbing.
    """

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    _SUFFIX_BY_MODE = {
        "myelinated": axonmyelin_suffix,
        "unmyelinated": unmyelinated_suffix,
        "nerve": nerve_suffix,
    }
    _INDEX_SUFFIX_BY_MODE = {
        "myelinated": axonmyelin_index_suffix,
        "unmyelinated": unmyelinated_index_suffix,
        "nerve": nerve_index_suffix,
    }

    def __init__(self):
        super().__init__()
        self.paths: List[str] = []
        self.axon_shape = "circle"
        self.pixel_size: Optional[float] = None  # None => auto-read pixel_size_in_micrometer.txt
        self.mode = "myelinated"  # myelinated | unmyelinated | nerve
        self.out_name = "axon_morphometrics.xlsx"
        self.colorize = False
        self.diameter_overlay = False
        self.cancel_requested = False

    def run(self):
        target_suffix = self._SUFFIX_BY_MODE[self.mode]
        target_index_suffix = self._INDEX_SUFFIX_BY_MODE[self.mode]

        targets = self._collect_targets(target_suffix)
        if not targets:
            self.log.emit(f"No image with a matching {target_suffix} mask found.")
            self.finished_ok.emit(False)
            return

        total = len(targets)
        n_ok = 0
        self.progress.emit(0, total, "Computing morphometrics")
        for i, img_path in enumerate(targets):
            if self.cancel_requested:
                self.log.emit("Morphometrics cancelled.")
                break
            try:
                self._process_one(img_path, target_suffix, target_index_suffix)
                n_ok += 1
            except Exception as e:
                self.log.emit(f"Failed on {img_path.name}: {e}")
            self.progress.emit(i + 1, total, "Computing morphometrics")

        self.log.emit(f"Morphometrics done for {n_ok}/{total} image(s).")
        self.finished_ok.emit(n_ok > 0)

    def _collect_targets(self, target_suffix: Path) -> List[Path]:
        targets = []
        seen = set()
        for p in self.paths:
            path = Path(p)
            candidates = sorted(path.iterdir()) if path.is_dir() else ([path] if path.exists() else [])
            for f in candidates:
                if f.suffix.lower() not in valid_extensions or str(f).endswith(generated_file_suffixes):
                    continue
                resolved = resolve_segmented_image_path(f)
                if not (resolved.parent / (resolved.stem + str(target_suffix))).exists():
                    continue
                if resolved not in seen:
                    targets.append(resolved)
                    seen.add(resolved)
        return targets

    def _resolve_pixel_size(self, img_path: Path) -> float:
        if self.pixel_size:
            return self.pixel_size
        pf = img_path.parent / "pixel_size_in_micrometer.txt"
        if pf.exists():
            return float(pf.read_text().strip())
        raise ValueError(
            "No pixel size given and no pixel_size_in_micrometer.txt in the image folder."
        )

    def _load_mask(self, img_path: Path, suffix: Path):
        mask_path = img_path.parent / (img_path.stem + str(suffix))
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask {mask_path.name} for {img_path.name}.")
        return ads.imread(str(mask_path))

    def _process_one(self, img_path: Path, target_suffix: Path, target_index_suffix: Path):
        import numpy as np

        psm = self._resolve_pixel_size(img_path)
        colorize = self.colorize and self.mode == "myelinated"
        diameter_overlay = self.diameter_overlay and self.mode == "myelinated"

        if self.mode == "myelinated":
            arg1 = self._load_mask(img_path, axon_suffix)
            arg2 = self._load_mask(img_path, myelin_suffix)
        elif self.mode == "unmyelinated":
            arg1 = self._load_mask(img_path, unmyelinated_suffix)
            arg2 = None
        else:  # nerve
            pred_nerve = self._load_mask(img_path, nerve_suffix)
            arg1 = self._load_mask(img_path, axon_suffix)
            arg2 = self._load_mask(img_path, myelin_suffix)

        morph_output = get_axon_morphometrics(
            im_axon=arg1,
            im_myelin=arg2,
            pixel_size=psm,
            axon_shape=self.axon_shape,
            return_index_image=True,
            return_border_info=True,
            return_instance_seg=colorize,
            return_im_axonmyelin_label=colorize,
        )
        stats_dataframe, index_image_array = morph_output[0:2]
        if colorize:
            instance_seg_image, instance_map = morph_output[2:]

        out_name = self.out_name or "axon_morphometrics.xlsx"
        if self.mode == "unmyelinated" and out_name == "axon_morphometrics.xlsx":
            out_name = "uaxon_morphometrics.xlsx"
        elif self.mode == "nerve" and out_name == "axon_morphometrics.xlsx":
            out_name = "nerve_morphometrics.json"
        if not out_name.lower().endswith((".xlsx", ".csv", ".json")):
            out_name += ".xlsx"

        morph_filename = img_path.stem + "_" + out_name

        if self.mode == "nerve":
            save_nerve_morphometrics_to_json(stats_dataframe, img_path.parent / morph_filename)
        else:
            save_axon_morphometrics(img_path.parent / morph_filename, stats_dataframe)

        outfile_basename = str(img_path.with_suffix(""))
        bg_image_path = str(img_path.with_suffix("")) + str(target_suffix)
        index_overlay_fname = outfile_basename + str(target_index_suffix)
        postprocessing.generate_and_save_colored_image_with_index_numbers(
            filename=index_overlay_fname,
            axonmyelin_image_path=bg_image_path,
            index_image_array=index_image_array,
        )

        if colorize:
            ads.imwrite(outfile_basename + str(instance_im_suffix), instance_seg_image)
            instance_map = instance_map.astype(np.uint16)
            ads.imwrite(outfile_basename + str(instance_suffix), instance_map, use_16bit=True)

        if diameter_overlay:
            overlay_array = postprocessing.generate_diameter_overlay(
                stats_dataframe, arg1.shape, psm, axon_shape=self.axon_shape
            )
            if overlay_array is not None:
                postprocessing.save_diameter_overlay(
                    overlay_array, outfile_basename + str(diameter_overlay_suffix)
                )

        if self.mode == "nerve":
            new_pred_axon, new_pred_myelin = remove_outside_nerve(arg1, arg2, pred_nerve)
            morph_output = get_axon_morphometrics(
                im_axon=new_pred_axon,
                im_myelin=new_pred_myelin,
                pixel_size=psm,
                axon_shape=self.axon_shape,
                return_index_image=True,
                return_border_info=True,
            )
            axon_stats_dataframe = morph_output[0]
            axon_morph_filename = img_path.stem + "_axon_morphometrics.xlsx"
            axon_morph_path = img_path.parent / axon_morph_filename
            save_axon_morphometrics(axon_morph_path, axon_stats_dataframe)

            nerve_morph_path = img_path.parent / morph_filename
            nerve_seg_path = img_path.parent / (img_path.stem + str(nerve_suffix))
            compute_axon_density(axon_morph_path, nerve_morph_path, nerve_seg_path)


class DownloadThread(QThread):
    """Downloads a single model variant."""

    log = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.model_key: str = ""
        self.variant: str = "light"  # light | ensemble

    def run(self):
        try:
            self.log.emit(f"Downloading {self.model_key} ({self.variant})...")
            download_model(self.model_key, destination=None, overwrite=True, variant=self.variant)
            self.log.emit(f"{self.model_key} ({self.variant}) downloaded.")
            self.finished_ok.emit(True)
        except Exception as e:
            self.log.emit(f"Download failed: {e}")
            self.finished_ok.emit(False)
