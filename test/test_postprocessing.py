# coding: utf-8

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from ads_base import ads_utils
from ads_base import postprocessing

class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent

        self.test_files_path = (self.fullPath / '__test_files__' / '__test_postprocessing_files__')
        print(self.test_files_path)

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
    def test_remove_axons_at_coordinates_returns_expected_masks(self):
        expected_image = ads_utils.imread((self.test_files_path / 'after_removing_two.png'))
        expected_axon, expected_myelin = ads_utils.extract_axon_and_myelin_masks_from_image_data(expected_image)
        im_axon, im_myelin = ads_utils.extract_axon_and_myelin_masks_from_image_data(self.before_axon_removal_image)
        obtained_axon, obtained_myelin = \
            postprocessing.remove_axons_at_coordinates(im_axon, im_myelin, [14.5, 42.5], [12.5, 12.5])

        assert (np.array_equal(expected_axon, obtained_axon)) \
               and \
               (np.array_equal(expected_myelin, obtained_myelin))
