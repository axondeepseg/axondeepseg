"""Unit tests for the standalone GUI's backend helpers.

Everything here is pure path/array logic — no model, no GPU, no real images — so
these run in milliseconds and don't need the downloaded test data.
"""
from pathlib import Path

import numpy as np
import pytest

from AxonDeepSeg.ads_gui.backend import (
    MorphometricsThread,
    SegmentThread,
    compose_overlay,
    expand_to_image_files,
    find_result_masks,
    resolve_segmented_image_path,
)


def _touch(*paths):
    """Create empty files. These helpers only ever stat/glob, never decode."""
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


class TestExpandToImageFiles:
    @pytest.mark.unit
    def test_folder_expands_to_raw_images_only(self, tmp_path):
        """Previously-generated masks/overlays must never be queued for segmentation."""
        _touch(
            tmp_path / "image.png",
            tmp_path / "other.tif",
            tmp_path / "image_seg-axon.png",
            tmp_path / "image_seg-myelin.png",
            tmp_path / "image_seg-axonmyelin.png",
            tmp_path / "image_seg-uaxon.png",
            tmp_path / "image_seg-nnunet.png",
            tmp_path / "image_index.png",
            tmp_path / "image_diameter_overlay.png",
        )

        assert expand_to_image_files([str(tmp_path)]) == [
            tmp_path / "image.png",
            tmp_path / "other.tif",
        ]

    @pytest.mark.unit
    def test_grayscale_copies_are_excluded(self, tmp_path):
        """segment.py writes a '<stem>_grayscale.png' copy; queueing it would re-segment
        the same image twice under a different name."""
        _touch(tmp_path / "image.png", tmp_path / "image_grayscale.png")

        assert expand_to_image_files([str(tmp_path)]) == [tmp_path / "image.png"]

    @pytest.mark.unit
    def test_non_image_files_are_ignored(self, tmp_path):
        _touch(
            tmp_path / "image.png",
            tmp_path / "notes.txt",
            tmp_path / "pixel_size_in_micrometer.txt",
            tmp_path / "axon_morphometrics.xlsx",
        )

        assert expand_to_image_files([str(tmp_path)]) == [tmp_path / "image.png"]

    @pytest.mark.unit
    def test_mixed_files_and_folders_are_deduplicated(self, tmp_path):
        """A user can add a file individually and then add its parent folder."""
        img = tmp_path / "image.png"
        _touch(img, tmp_path / "second.png")

        result = expand_to_image_files([str(img), str(tmp_path)])

        assert result == [img, tmp_path / "second.png"]

    @pytest.mark.unit
    def test_missing_paths_are_skipped(self, tmp_path):
        img = tmp_path / "image.png"
        _touch(img)

        assert expand_to_image_files([str(tmp_path / "gone.png"), str(img)]) == [img]

    @pytest.mark.unit
    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert expand_to_image_files([str(tmp_path)]) == []


class TestResolveSegmentedImagePath:
    @pytest.mark.unit
    def test_prefers_grayscale_copy_when_present(self, tmp_path):
        """Masks are written next to the grayscale copy, so morphometrics has to
        follow the rename rather than look beside the original file."""
        original = tmp_path / "image.tif"
        grayscale = tmp_path / "image_grayscale.png"
        _touch(original, grayscale)

        assert resolve_segmented_image_path(original) == grayscale

    @pytest.mark.unit
    def test_falls_back_to_original_when_no_grayscale_copy(self, tmp_path):
        original = tmp_path / "image.png"
        _touch(original)

        assert resolve_segmented_image_path(original) == original


class TestFindResultMasks:
    @pytest.mark.unit
    def test_finds_myelinated_pair(self, tmp_path):
        img = tmp_path / "image.png"
        _touch(img, tmp_path / "image_seg-axon.png", tmp_path / "image_seg-myelin.png")

        masks = find_result_masks(img)

        assert masks["axon"] == tmp_path / "image_seg-axon.png"
        assert masks["myelin"] == tmp_path / "image_seg-myelin.png"
        assert masks["uaxon"] is None

    @pytest.mark.unit
    def test_finds_unmyelinated_mask_independently(self, tmp_path):
        """Some models output only unmyelinated axons, others output both classes —
        these are checked independently rather than as an either/or."""
        img = tmp_path / "image.png"
        _touch(img, tmp_path / "image_seg-uaxon.png")

        masks = find_result_masks(img)

        assert masks["uaxon"] == tmp_path / "image_seg-uaxon.png"
        assert masks["axon"] is None
        assert masks["myelin"] is None

    @pytest.mark.unit
    def test_returns_all_none_when_nothing_was_segmented(self, tmp_path):
        img = tmp_path / "image.png"
        _touch(img)

        assert find_result_masks(img) == {"axon": None, "myelin": None, "uaxon": None}


class TestComposeOverlay:
    @staticmethod
    def _gray(value=100, shape=(4, 4)):
        return np.full(shape, value, dtype=np.uint8)

    @pytest.mark.unit
    def test_returns_rgb_uint8(self):
        out = compose_overlay(self._gray())

        assert out.shape == (4, 4, 3)
        assert out.dtype == np.uint8

    @pytest.mark.unit
    def test_unmasked_pixels_keep_the_original_grey(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 255

        out = compose_overlay(self._gray(100), axon=mask)

        assert tuple(out[3, 3]) == (100, 100, 100)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwarg, dominant",
        [("axon", 2), ("myelin", 0), ("uaxon", 1)],
    )
    def test_each_class_tints_its_own_channel(self, kwarg, dominant):
        """axon is blue, myelin is red, unmyelinated axon is green — matching the
        napari plugin's mask colors."""
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 255

        out = compose_overlay(self._gray(100), **{kwarg: mask})

        pixel = out[0, 0]
        others = [c for i, c in enumerate(pixel) if i != dominant]
        assert pixel[dominant] > max(others)

    @pytest.mark.unit
    @pytest.mark.parametrize("on_value", [1, 255])
    def test_handles_both_binary_and_8bit_masks(self, on_value):
        """Masks reach the GUI as either 0/1 or 0/255 depending on which code path
        wrote them, so the threshold is derived from the mask's own max."""
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = on_value

        out = compose_overlay(self._gray(100), axon=mask)

        assert out[0, 0][2] > 100, "masked pixel should be tinted blue"
        assert tuple(out[3, 3]) == (100, 100, 100), "unmasked pixel should be untouched"

    @pytest.mark.unit
    def test_all_zero_mask_tints_nothing(self):
        out = compose_overlay(self._gray(100), axon=np.zeros((4, 4), dtype=np.uint8))

        assert np.all(out == 100)

    @pytest.mark.unit
    def test_axon_is_drawn_over_myelin_where_they_overlap(self):
        both = np.full((4, 4), 255, dtype=np.uint8)

        out = compose_overlay(self._gray(100), axon=both, myelin=both)

        pixel = out[0, 0]
        assert pixel[2] > pixel[0], "axon (blue) should win over myelin (red)"

    @pytest.mark.unit
    def test_rgb_input_is_reduced_to_one_channel(self):
        rgb = np.dstack([self._gray(100), self._gray(50), self._gray(20)])

        out = compose_overlay(rgb)

        assert out.shape == (4, 4, 3)
        assert tuple(out[0, 0]) == (100, 100, 100)


class TestSegmentThreadGpuSelection:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "available, expected",
        [(0, -1), (1, 0), (4, 0)],
    )
    def test_picks_gpu_zero_only_when_one_exists(self, monkeypatch, available, expected):
        monkeypatch.setattr(
            "AxonDeepSeg.ads_gui.backend.ads.check_available_gpus",
            lambda _: available,
        )

        assert SegmentThread._pick_gpu() == expected

    @pytest.mark.unit
    def test_falls_back_to_cpu_when_detection_raises(self, monkeypatch):
        """A broken/absent CUDA install shouldn't stop the user from segmenting."""
        def _boom(_):
            raise RuntimeError("no cuda")

        monkeypatch.setattr(
            "AxonDeepSeg.ads_gui.backend.ads.check_available_gpus", _boom
        )

        assert SegmentThread._pick_gpu() == -1


class TestMorphometricsThreadTargets:
    @pytest.mark.unit
    def test_only_images_with_a_matching_mask_are_collected(self, qapp, tmp_path):
        segmented = tmp_path / "segmented.png"
        _touch(
            segmented,
            tmp_path / "segmented_seg-axonmyelin.png",
            tmp_path / "not_segmented.png",
        )

        thread = MorphometricsThread()
        thread.paths = [str(tmp_path)]

        assert thread._collect_targets(Path("_seg-axonmyelin.png")) == [segmented]

    @pytest.mark.unit
    def test_targets_resolve_through_the_grayscale_rename(self, qapp, tmp_path):
        _touch(
            tmp_path / "image.tif",
            tmp_path / "image_grayscale.png",
            tmp_path / "image_grayscale_seg-axonmyelin.png",
        )

        thread = MorphometricsThread()
        thread.paths = [str(tmp_path)]

        assert thread._collect_targets(Path("_seg-axonmyelin.png")) == [
            tmp_path / "image_grayscale.png"
        ]

    @pytest.mark.unit
    def test_same_image_added_twice_is_collected_once(self, qapp, tmp_path):
        img = tmp_path / "image.png"
        _touch(img, tmp_path / "image_seg-axonmyelin.png")

        thread = MorphometricsThread()
        thread.paths = [str(img), str(tmp_path)]

        assert thread._collect_targets(Path("_seg-axonmyelin.png")) == [img]


class TestMorphometricsThreadPixelSize:
    @pytest.mark.unit
    def test_explicit_pixel_size_wins(self, qapp, tmp_path):
        (tmp_path / "pixel_size_in_micrometer.txt").write_text("0.99")

        thread = MorphometricsThread()
        thread.pixel_size = 0.07

        assert thread._resolve_pixel_size(tmp_path / "image.png") == 0.07

    @pytest.mark.unit
    def test_falls_back_to_the_sidecar_file(self, qapp, tmp_path):
        (tmp_path / "pixel_size_in_micrometer.txt").write_text("0.07\n")

        thread = MorphometricsThread()
        thread.pixel_size = None

        assert thread._resolve_pixel_size(tmp_path / "image.png") == 0.07

    @pytest.mark.unit
    def test_raises_when_pixel_size_is_unknown(self, qapp, tmp_path):
        thread = MorphometricsThread()
        thread.pixel_size = None

        with pytest.raises(ValueError, match="No pixel size"):
            thread._resolve_pixel_size(tmp_path / "image.png")
