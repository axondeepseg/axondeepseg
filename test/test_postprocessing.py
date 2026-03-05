# coding: utf-8

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from skimage.measure import regionprops, label

from AxonDeepSeg import ads_utils
from AxonDeepSeg import postprocessing

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent

        self.test_files_path = (self.fullPath / '__test_files__' / '__test_postprocessing_files__')

        self.before_floodfill_image = ads_utils.imread((self.test_files_path / 'before_flood_fill.png'))
        self.after_floodfill_image = ads_utils.imread((self.test_files_path / 'after_flood_fill.png'))

        self.before_axon_removal_image = ads_utils.imread((self.test_files_path / 'before_removing.png'))

    def teardown_method(self):
        pass

    @pytest.mark.unit
    def test_get_centroids_returns_expected_list(self):
        axon_mask, _ = ads_utils.extract_axon_and_myelin_masks_from_image_data(self.before_floodfill_image)
        expected_centroids = ([5, 11, 41, 55], [5, 53, 39, 53])
        obtained_centroids = postprocessing.get_centroids(axon_mask)
        assert sorted(expected_centroids) == sorted(obtained_centroids)

    @pytest.mark.unit
    def test_fill_myelin_holes_returns_expected_arrays(self):
        # We can reuse the data from the floodfill since the goal of these two tools is the same
        _, before_fill_myelin_mask = \
            ads_utils.extract_axon_and_myelin_masks_from_image_data(self.before_floodfill_image)
        after_fill_expected_axon_mask, _ = \
            ads_utils.extract_axon_and_myelin_masks_from_image_data(self.after_floodfill_image)
        after_fill_obtained_axon_mask = postprocessing.fill_myelin_holes(before_fill_myelin_mask)

        assert np.array_equal(after_fill_obtained_axon_mask, after_fill_expected_axon_mask)

    @pytest.mark.unit
    def test_remove_intersection_returns_expected_arrays(self):
        mask1_in = np.zeros((3, 3))
        mask1_in[:, 0:2] = 1

        mask2_in = np.zeros((3, 3))
        mask2_in[:, 1:3] = 1

        mask1_expected_out = np.zeros((3, 3))
        mask1_expected_out[:, 0] = 1

        mask2_expected_out = np.array(mask2_in, copy=True)

        expected_overlap = np.zeros((3, 3))
        expected_overlap[:, 1] = 1

        mask1_obtained_out, mask2_obtained_out, obtained_overlap = \
            postprocessing.remove_intersection(mask1_in, mask2_in, priority=2, return_overlap=True)

        expectations = [mask1_expected_out, mask2_expected_out, expected_overlap]
        obtained_results = [mask1_expected_out, mask2_expected_out, obtained_overlap]

        for i in range(expectations.__len__()):
            assert np.array_equal(expectations[i], obtained_results[i])

    @pytest.mark.unit
    def test_generate_axon_numbers_image_returns_expected_array(self):
        # Load the test image
        expected_image = ads_utils.imread((self.test_files_path / 'test_numbers_image.png'))
        expected_image = np.array(expected_image)

        # Atempt to recreate the test image
        obtained_image = postprocessing.generate_axon_numbers_image(
            centroid_index=np.array([0]),
            x0_array=[20],
            y0_array=[10],
            image_size=(30, 30),
            mean_axon_diameter_in_pixels=6
        )
        assert np.array_equal(expected_image, obtained_image)

    @pytest.mark.unit
    def test_generate_and_save_colored_image_with_index_numbers_saves_the_correct_image(self):
        # Load the test images/variables
        expected_image = np.asarray(Image.open(self.test_files_path / "test_axonmyelin_index.png"))
        output_image_path = self.test_files_path / "output_axonmyelin_index.png"
        axonmyelin_image_path = self.test_files_path / "test_axonmyelin.png"
        axonmyelin_image = ads_utils.imread(axonmyelin_image_path)
        # Attempt to recreate the test image
        index_array =  postprocessing.generate_axon_numbers_image(
            centroid_index=np.array([0]),
            x0_array=[axonmyelin_image.shape[1]//2],
            y0_array=[axonmyelin_image.shape[0]//2],
            image_size=axonmyelin_image.shape
        )
        postprocessing.generate_and_save_colored_image_with_index_numbers(
            filename=output_image_path,
            axonmyelin_image_path=axonmyelin_image_path,
            index_image_array=index_array
        )
        # Load the created image and compare it to the expected image
        output_image = np.asarray(Image.open(output_image_path))
        assert np.array_equal(output_image, expected_image)

    @pytest.mark.unit
    def test_generate_rotated_ellipse_points_returns_requested_number_of_points(self):
        points = postprocessing.generate_rotated_ellipse_points(
            center_x=100, center_y=100, semi_major=20, semi_minor=10, orientation=0, num_points=64
        )
        assert len(points) == 64

    @pytest.mark.unit
    def test_generate_rotated_ellipse_points_circle_has_constant_radius(self):
        """When semi_major == semi_minor, all points must be equidistant from the center."""
        cx, cy, r = 50.0, 80.0, 15.0
        points = postprocessing.generate_rotated_ellipse_points(
            center_x=cx, center_y=cy, semi_major=r, semi_minor=r, orientation=0, num_points=128
        )
        distances = [np.hypot(x - cx, y - cy) for x, y in points]
        assert np.allclose(distances, r, atol=1e-10)

    @pytest.mark.unit
    def test_generate_rotated_ellipse_points_extremal_points_match_axes(self):
        """With orientation=0, the farthest point from center should be ~semi_major away."""
        cx, cy = 0.0, 0.0
        semi_major, semi_minor = 30.0, 10.0
        points = postprocessing.generate_rotated_ellipse_points(
            center_x=cx, center_y=cy, semi_major=semi_major, semi_minor=semi_minor,
            orientation=0, num_points=1000
        )
        distances = [np.hypot(x - cx, y - cy) for x, y in points]
        assert np.isclose(max(distances), semi_major, atol=1e-2)
        assert np.isclose(min(distances), semi_minor, atol=1e-2)

    @pytest.mark.unit
    def test_generate_diameter_overlay_circle_enclosed_area_matches_axon_morphometric(self):
        """
        In circle mode, axon_diam = equivalent_diameter = sqrt(4*area/π), so the circle
        drawn by the overlay encloses the same area as the axon mask.
        Count enclosed pixels by checking that each non-outline pixel has outline pixels
        on both sides in both x and y directions.
        """
        pixel_size = 0.1  # µm/px

        # 50×50 square axon centered in a 200×200 image
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[75:125, 75:125] = 255
        mask_area_px = int((mask > 0).sum())

        # Derive axon_diam exactly as get_axon_morphometrics does in circle mode
        props = regionprops(label(mask > 0))[0]
        axon_diam_um = props.equivalent_diameter_area * pixel_size   # µm
        cy, cx = props.centroid                                       # (row, col) → (y0, x0)

        df = pd.DataFrame([{
            'x0': cx, 'y0': cy,
            'axon_diam': axon_diam_um,
            'myelin_thickness': np.nan,
        }])
        overlay = postprocessing.generate_diameter_overlay(
            df, image_shape=(200, 200), pixel_size=pixel_size, axon_shape='circle'
        )

        # A non-outline pixel is inside the circle if, in both x and y, there are
        # outline pixels on both sides of it.
        outline = overlay == 255
        row_has_left  = np.cumsum(outline, axis=1) > 0
        row_has_right = np.cumsum(outline[:, ::-1], axis=1)[:, ::-1] > 0
        col_has_above = np.cumsum(outline, axis=0) > 0
        col_has_below = np.cumsum(outline[::-1, :], axis=0)[::-1, :] > 0

        inside = (~outline) & row_has_left & row_has_right & col_has_above & col_has_below
        enclosed_area_px = int(inside.sum()) + int(outline.sum())

        assert np.isclose(enclosed_area_px, mask_area_px, rtol=0.05)

    @pytest.mark.unit
    def test_generate_diameter_overlay_circle_inner_and_outer_are_concentric(self):
        """Inner and outer rings must share the same center."""
        pixel_size = 0.1
        axon_diam = 10.0      # µm → inner radius = 50 px
        myelin_thickness = 3.0  # µm → outer radius = 80 px
        cx, cy = 200, 150
        inner_radius_px = axon_diam / (2 * pixel_size)
        outer_radius_px = inner_radius_px + myelin_thickness / pixel_size

        df = pd.DataFrame([{
            'x0': cx, 'y0': cy,
            'axon_diam': axon_diam,
            'myelin_thickness': myelin_thickness,
        }])
        overlay = postprocessing.generate_diameter_overlay(
            df, image_shape=(400, 400), pixel_size=pixel_size, axon_shape='circle'
        )

        ys, xs = np.where(overlay > 0)
        distances = np.hypot(xs - cx, ys - cy)
        midpoint = (inner_radius_px + outer_radius_px) / 2
        inner_mask = distances < midpoint
        outer_mask = ~inner_mask

        inner_center_x = np.mean(xs[inner_mask])
        inner_center_y = np.mean(ys[inner_mask])
        outer_center_x = np.mean(xs[outer_mask])
        outer_center_y = np.mean(ys[outer_mask])

        assert np.isclose(inner_center_x, outer_center_x, atol=1.0)
        assert np.isclose(inner_center_y, outer_center_y, atol=1.0)

    @pytest.mark.unit
    def test_generate_diameter_overlay_ellipse_minor_axis_gap_matches_myelin_thickness(self):
        """
        In ellipse mode with orientation=0, the polygon vertex at index 0 falls exactly
        at (cx + semi_minor, cy). The gap between the inner and outer outline centroids
        along that horizontal line should equal myelin_thickness / pixel_size.
        """
        pixel_size = 0.1
        axon_diam = 20.0        # µm → axon_radius_px = 100 px
        myelin_thickness = 5.0  # µm → 50 px gap along minor axis
        cx, cy = 300, 300
        axon_radius_px = axon_diam / (2 * pixel_size)       # 100
        myelin_thickness_px = myelin_thickness / pixel_size  # 50

        df = pd.DataFrame([{
            'x0': cx, 'y0': cy,
            'axon_diam': axon_diam,
            'myelin_thickness': myelin_thickness,
            'eccentricity': 0.6,
            'orientation': 0.0,
            'fiber_eccentricity': 0.5,
            'fiber_orientation': 0.0,
        }])
        overlay = postprocessing.generate_diameter_overlay(
            df, image_shape=(600, 600), pixel_size=pixel_size, axon_shape='ellipse'
        )

        # With orientation=0, the first polygon vertex for each ellipse is placed at
        # (cx + semi_minor, cy). Scan a ±3-pixel horizontal strip around cy for robustness.
        strip_max = overlay[cy - 3:cy + 4, :].max(axis=0)
        all_cols = np.arange(600)
        right_lit_cols = all_cols[(strip_max > 0) & (all_cols > cx)]

        # Split into inner cluster (≈cx+axon_radius_px) and outer (≈cx+outer_semi_minor)
        midpoint_col = cx + axon_radius_px + myelin_thickness_px / 2
        inner_cols = right_lit_cols[right_lit_cols < midpoint_col]
        outer_cols = right_lit_cols[right_lit_cols >= midpoint_col]

        gap_px = np.mean(outer_cols) - np.mean(inner_cols)
        assert np.isclose(gap_px, myelin_thickness_px, atol=3.0)

    @pytest.mark.unit
    def test_generate_diameter_overlay_ellipse_equals_circle_enclosed_area_for_perfect_circle(self):
        """
        For a perfect circle (eccentricity=0), semi_major == semi_minor, so the ellipse
        mode overlay is identical in shape to the circle mode overlay.
        The total enclosed area (interior + outline) must agree within 2%.
        """
        pixel_size = 0.1
        axon_diam = 10.0      # µm → radius = 50 px
        myelin_thickness = 2.0  # µm → 20 px ring
        cx, cy = 200, 200

        base = {'x0': cx, 'y0': cy, 'axon_diam': axon_diam, 'myelin_thickness': myelin_thickness}

        overlay_circle = postprocessing.generate_diameter_overlay(
            pd.DataFrame([base]), image_shape=(400, 400), pixel_size=pixel_size, axon_shape='circle'
        )
        overlay_ellipse = postprocessing.generate_diameter_overlay(
            pd.DataFrame([{**base, 'eccentricity': 0.0, 'orientation': 0.0,
                           'fiber_eccentricity': 0.0, 'fiber_orientation': 0.0}]),
            image_shape=(400, 400), pixel_size=pixel_size, axon_shape='ellipse'
        )

        def count_enclosed(overlay):
            outline = overlay == 255
            inside = (
                (~outline)
                & (np.cumsum(outline, axis=1) > 0)
                & (np.cumsum(outline[:, ::-1], axis=1)[:, ::-1] > 0)
                & (np.cumsum(outline, axis=0) > 0)
                & (np.cumsum(outline[::-1, :], axis=0)[::-1, :] > 0)
            )
            return int(inside.sum()) + int(outline.sum())

        assert np.isclose(count_enclosed(overlay_circle), count_enclosed(overlay_ellipse), rtol=0.02)

    @pytest.mark.unit
    def test_generate_diameter_overlay_ellipse_ring_gap_equals_circle_ring_gap_for_perfect_circle(self):
        """
        For a perfect circle (eccentricity=0), the ring gap measured from the ellipse
        mode overlay should equal the gap from the circle mode overlay.
        """
        pixel_size = 0.1
        axon_diam = 10.0      # µm → axon radius = 50 px
        myelin_thickness = 2.0  # µm → 20 px ring
        cx, cy = 200, 200
        axon_radius_px = axon_diam / (2 * pixel_size)
        myelin_thickness_px = myelin_thickness / pixel_size

        base = {'x0': cx, 'y0': cy, 'axon_diam': axon_diam, 'myelin_thickness': myelin_thickness}

        overlay_circle = postprocessing.generate_diameter_overlay(
            pd.DataFrame([base]), image_shape=(400, 400), pixel_size=pixel_size, axon_shape='circle'
        )
        overlay_ellipse = postprocessing.generate_diameter_overlay(
            pd.DataFrame([{**base, 'eccentricity': 0.0, 'orientation': 0.0,
                           'fiber_eccentricity': 0.0, 'fiber_orientation': 0.0}]),
            image_shape=(400, 400), pixel_size=pixel_size, axon_shape='ellipse'
        )

        def measure_right_gap(overlay):
            # Scan a ±3-row strip at cy; split lit pixels right of cx into inner/outer clusters
            strip_max = overlay[cy - 3:cy + 4, :].max(axis=0)
            all_cols = np.arange(400)
            right_lit = all_cols[(strip_max > 0) & (all_cols > cx)]
            midpoint = cx + axon_radius_px + myelin_thickness_px / 2
            return np.mean(right_lit[right_lit >= midpoint]) - np.mean(right_lit[right_lit < midpoint])

        assert np.isclose(measure_right_gap(overlay_circle), measure_right_gap(overlay_ellipse), atol=3.0)

    @pytest.mark.unit
    def test_save_diameter_overlay_does_not_crash_and_creates_no_file_when_given_none(self, tmp_path):
        output_path = tmp_path / 'overlay.png'
        postprocessing.save_diameter_overlay(None, output_path)
        assert not output_path.exists()

    @pytest.mark.unit
    def test_save_diameter_overlay_writes_file_for_valid_array(self, tmp_path):
        overlay = np.zeros((100, 100), dtype=np.uint8)
        overlay[50, 50] = 255
        output_path = tmp_path / 'overlay.png'
        postprocessing.save_diameter_overlay(overlay, output_path)
        assert output_path.exists()

    @pytest.mark.unit
    def test_remove_axons_at_coordinates_returns_expected_masks(self):
        expected_image = ads_utils.imread((self.test_files_path / 'after_removing_two.png'))
        expected_axon, expected_myelin = ads_utils.extract_axon_and_myelin_masks_from_image_data(expected_image)
        im_axon, im_myelin = ads_utils.extract_axon_and_myelin_masks_from_image_data(self.before_axon_removal_image)
        obtained_axon, obtained_myelin = \
            postprocessing.remove_axons_at_coordinates(im_axon, im_myelin, [14.5, 42.5], [12.5, 12.5])

        assert (np.array_equal(expected_axon, obtained_axon)) \
               and \
               (np.array_equal(expected_myelin, obtained_myelin))
