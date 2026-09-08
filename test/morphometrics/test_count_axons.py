# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AxonDeepSeg.ads_utils import imwrite
from AxonDeepSeg.morphometrics.count_axons import main


class TestCore(object):
    def setup_method(self):
        self.fullPath = Path(__file__).resolve().parent
        self.testPath = self.fullPath.parent

    def teardown_method(self):
        pass

    # --------------main integration tests-------------- #
    @pytest.mark.integration
    def test_main_counts_morphometric_rows_and_prefers_filtered_files(self, tmp_path):
        image_path = tmp_path / 'sample.png'
        image_path.touch()

        pd.DataFrame({'axon_diam (um)': [1, 2]}).to_excel(
            tmp_path / 'sample_axon_morphometrics.xlsx',
            index=False,
        )
        pd.DataFrame({'axon_diam (um)': [1]}).to_excel(
            tmp_path / 'sample_axon_morphometrics_filtered.xlsx',
            index=False,
        )
        pd.DataFrame({'axon_diam (um)': [1, 2, 3]}).to_excel(
            tmp_path / 'sample_uaxon_morphometrics.xlsx',
            index=False,
        )

        output_path = tmp_path / 'counts.xlsx.csv'
        main(['-i', str(tmp_path), '-o', str(output_path)])

        result = pd.read_csv(output_path)

        assert result.to_dict('records') == [{
            'image': 'sample',
            'axon_count': 1,
            'uaxon_count': 3,
        }]

    @pytest.mark.integration
    def test_main_counts_connected_components_and_prefers_filtered_masks(self, tmp_path):
        image = np.zeros((10, 10), dtype=np.uint8)
        axonmyelin = np.zeros((10, 10), dtype=np.uint8)
        uaxon = np.zeros((10, 10), dtype=np.uint8)
        filtered_axonmyelin = np.zeros((10, 10), dtype=np.uint8)
        filtered_uaxon = np.zeros((10, 10), dtype=np.uint8)

        axonmyelin[1:3, 1:3] = 255
        axonmyelin[7:9, 7:9] = 255
        uaxon[1:3, 1:3] = 255
        uaxon[7:9, 7:9] = 255
        filtered_axonmyelin[1:3, 1:3] = 255
        filtered_uaxon[1:3, 1:3] = 255

        imwrite(tmp_path / 'sample.png', image)
        imwrite(tmp_path / 'sample_seg-axonmyelin.png', axonmyelin)
        imwrite(tmp_path / 'sample_seg-uaxon.png', uaxon)
        imwrite(tmp_path / 'sample_seg-axonmyelin_filtered.png', filtered_axonmyelin)
        imwrite(tmp_path / 'sample_seg-uaxon_filtered.png', filtered_uaxon)

        output_path = tmp_path / 'counts.csv'
        main(['-i', str(tmp_path), '--mask_mode', '-o', str(output_path)])

        result = pd.read_csv(output_path)

        assert result.to_dict('records') == [{
            'image': 'sample',
            'axon_count': 1,
            'uaxon_count': 1,
        }]

    @pytest.mark.integration
    def test_main_writes_empty_counts_csv_for_empty_input_folder(self, tmp_path):
        output_path = tmp_path / 'counts.csv'

        main(['-i', str(tmp_path), '-o', str(output_path)])

        result = pd.read_csv(output_path)

        assert result.empty
        assert result.columns.tolist() == ['image', 'axon_count', 'uaxon_count']

    @pytest.mark.exceptionhandling
    def test_main_raises_for_missing_input_folder(self, tmp_path):
        with pytest.raises(AssertionError):
            main(['-i', str(tmp_path / 'missing')])
