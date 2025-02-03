from pathlib import Path
import pytest

import numpy as np
from skimage import measure

from ads_base.visualization.colorization import colorize_instance_segmentation, color_generator
from ads_base.visualization.simulate_axons import SimulateAxons, calc_myelin_thickness
from ads_base.morphometrics.compute_morphometrics import get_watershed_segmentation
from ads_base import ads_utils as adsutils


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent

        # simulated axons to test --border-info option
        self.simulated_image_path = Path(
            self.testPath /
            '__test_files__' /
            '__test_simulated_axons__' /
            'simulation_to_colorize.png'
        )

        simulated = SimulateAxons()
        # axon 1
        x_pos = 100
        y_pos = 100
        axon_radius = 100
        gratio = 0.6
        simulated.generate_axon(axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0)
        # axon 2
        x_pos = 500
        y_pos = 500
        axon_radius = 40
        gratio = 0.6
        simulated.generate_axon(axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0)
        # axon 3
        x_pos = 750
        y_pos = 750
        axon_radius = 40
        gratio = 0.6
        simulated.generate_axon(axon_radius=axon_radius, center=[x_pos, y_pos], gratio=gratio, plane_angle=0)

        simulated.save(self.simulated_image_path)

    def teardown_method(self):
        if self.simulated_image_path.is_file():
            self.simulated_image_path.unlink()

    @pytest.mark.unit
    def test_color_generator_outputs_different_colors(self):
        random_colors = [
            (0, 128, 128),
            (153, 50, 204),
            (0, 255, 127),
            (139, 0, 0),
        ]
        cg = color_generator(random_colors, 100, 5)
        generated_colors = np.array([])
        for i in range(100):
            np.append(generated_colors, next(cg))

        assert len(np.unique(generated_colors)) == len(generated_colors)

    @pytest.mark.unit
    def test_colorize_instance_segmentation_on_simulated_image(self):
        mask = adsutils.imread(self.simulated_image_path)
        axon, myelin = adsutils.extract_axon_and_myelin_masks_from_image_data(mask)

        im_axon_label = measure.label(axon)
        axon_objects = measure.regionprops(im_axon_label)
        ind_centroid = ([int(props.centroid[0]) for props in axon_objects],
                        [int(props.centroid[1]) for props in axon_objects])
        instance_seg = get_watershed_segmentation(axon, myelin, ind_centroid)
        colorized = colorize_instance_segmentation(instance_seg)

        # flatten the values
        flattened = np.dot(colorized[...,:3], [0.2989, 0.5870, 0.1140])

        # 3 axons + background should give us 4 pixel values
        assert len(np.unique(flattened)) == 4
