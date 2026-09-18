from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from AxonDeepSeg.morphometrics.filter_morphometrics import (
	apply_myelinated_rules,
	apply_unmyelinated_rules,
	main,
	read_config,
)


class TestCore(object):
	def setup_method(self):
		self.repository_path = Path(__file__).resolve().parents[2]
		self.default_config_path = (
			self.repository_path / 'AxonDeepSeg' / 'morphometrics' / 'filter.yaml'
		)

	# --------------read_config tests-------------- #
	@pytest.mark.unit
	def test_read_config_returns_expected_default_rules(self):
		config = read_config(self.default_config_path)

		assert set(config) == {'myelinated', 'unmyelinated'}
		assert config['myelinated'] == [
			{'valid-g-ratio-only': True},
			{'axon-diam-gt': None},
		]
		assert config['unmyelinated'] == [
			{'axon-diam-gt': None},
			{'solidity-gt': None},
			{'axon-area-lt': None},
		]

	@pytest.mark.unit
	def test_read_config_raises_for_missing_file(self, tmp_path):
		with pytest.raises(FileNotFoundError):
			read_config(tmp_path / 'missing.yaml')

	@pytest.mark.unit
	def test_read_config_raises_for_invalid_top_level_keys(self, tmp_path):
		config_path = tmp_path / 'invalid.yaml'
		config_path.write_text('myelinated: []\nother: []\n')

		with pytest.raises(ValueError):
			read_config(config_path)

	# --------------apply_myelinated_rules tests-------------- #
	@pytest.mark.unit
	def test_apply_myelinated_rules_keeps_only_valid_g_ratios(self):
		dataframe = pd.DataFrame({
			'gratio': [0.5, 0, 1, -0.1, np.nan, 0.9],
			'axon_diam (um)': [1, 2, 3, 4, 5, 6],
		})

		result = apply_myelinated_rules(
			dataframe,
			[{'valid-g-ratio-only': True}],
		)

		assert result.index.tolist() == [0, 5]

	@pytest.mark.unit
	def test_apply_myelinated_rules_uses_strict_diameter_threshold(self):
		dataframe = pd.DataFrame({
			'gratio': [0.5, 0.5, 0.5],
			'axon_diam (um)': [5, 10, 15],
		})

		result = apply_myelinated_rules(
			dataframe,
			[{'axon-diam-gt': 10}],
		)

		assert result.index.tolist() == [2]

	@pytest.mark.unit
	def test_apply_myelinated_rules_ignores_none_threshold(self):
		dataframe = pd.DataFrame({
			'gratio': [0.5, 0.8],
			'axon_diam (um)': [5, 10],
		})

		result = apply_myelinated_rules(
			dataframe,
			[{'axon-diam-gt': None}],
		)

		pd.testing.assert_frame_equal(result, dataframe)

	@pytest.mark.unit
	def test_apply_myelinated_rules_logs_unknown_rule(self):
		dataframe = pd.DataFrame({'gratio': [0.5]})

		with patch(
			'AxonDeepSeg.morphometrics.filter_morphometrics.logger.warning'
		) as warning:
			result = apply_myelinated_rules(dataframe, [{'unknown-rule': True}])

		pd.testing.assert_frame_equal(result, dataframe)
		warning.assert_called_once_with("Unknown rule: {'unknown-rule': True}")

	# --------------apply_unmyelinated_rules tests-------------- #
	@pytest.mark.unit
	def test_apply_unmyelinated_rules_applies_all_rules(self):
		dataframe = pd.DataFrame({
			'axon_diam (um)': [2, 6, 8],
			'solidity': [0.5, 0.8, 0.95],
			'axon_area (um^2)': [20, 10, 5],
		})

		result = apply_unmyelinated_rules(
			dataframe,
			[
				{'axon-diam-gt': 5},
				{'solidity-gt': 0.7},
				{'axon-area-lt': 11},
			],
		)

		assert result.index.tolist() == [1, 2]

	@pytest.mark.unit
	def test_apply_unmyelinated_rules_uses_strict_comparisons(self):
		dataframe = pd.DataFrame({
			'axon_diam (um)': [5, 6],
			'solidity': [0.7, 0.8],
			'axon_area (um^2)': [10, 9],
		})

		result = apply_unmyelinated_rules(
			dataframe,
			[
				{'axon-diam-gt': 5},
				{'solidity-gt': 0.7},
				{'axon-area-lt': 10},
			],
		)

		assert result.index.tolist() == [1]

	@pytest.mark.unit
	def test_apply_unmyelinated_rules_ignores_none_thresholds(self):
		dataframe = pd.DataFrame({
			'axon_diam (um)': [5],
			'solidity': [0.7],
			'axon_area (um^2)': [10],
		})

		result = apply_unmyelinated_rules(
			dataframe,
			[
				{'axon-diam-gt': None},
				{'solidity-gt': None},
				{'axon-area-lt': None},
			],
		)

		pd.testing.assert_frame_equal(result, dataframe)

	@pytest.mark.unit
	def test_apply_unmyelinated_rules_logs_unknown_rule(self):
		dataframe = pd.DataFrame({'axon_diam (um)': [5]})

		with patch(
			'AxonDeepSeg.morphometrics.filter_morphometrics.logger.warning'
		) as warning:
			result = apply_unmyelinated_rules(dataframe, [{'unknown-rule': True}])

		pd.testing.assert_frame_equal(result, dataframe)
		warning.assert_called_once_with("Unknown rule: {'unknown-rule': True}")

	# --------------main integration tests-------------- #
	@pytest.mark.integration
	def test_main_filters_myelinated_and_unmyelinated_files_in_folder(
		self, tmp_path
	):
		myelinated_file = tmp_path / 'sample_axon_morphometrics.xlsx'
		unmyelinated_file = tmp_path / 'sample_uaxon_morphometrics.xlsx'
		config_file = tmp_path / 'filter.yaml'

		pd.DataFrame({
			'Unnamed: 0': [0, 1, 2],
			'gratio': [0.5, 0, 1.1],
			'axon_diam (um)': [5, 10, 15],
		}).to_excel(myelinated_file, index=False)
		pd.DataFrame({
			'Unnamed: 0': [0, 1, 2],
			'axon_diam (um)': [2, 6, 8],
			'solidity': [0.5, 0.8, 0.95],
			'axon_area (um^2)': [20, 10, 5],
		}).to_excel(unmyelinated_file, index=False)
		config_file.write_text(
			'myelinated:\n'
			'  - valid-g-ratio-only: true\n'
			'  - axon-diam-gt: 4\n'
			'unmyelinated:\n'
			'  - axon-diam-gt: 5\n'
			'  - solidity-gt: 0.7\n'
			'  - axon-area-lt: 11\n'
		)

		main([
			'--input', str(tmp_path),
			'--config', str(config_file),
		])

		filtered_myelinated = pd.read_excel(
			tmp_path / 'sample_axon_morphometrics_filtered.xlsx'
		)
		filtered_unmyelinated = pd.read_excel(
			tmp_path / 'sample_uaxon_morphometrics_filtered.xlsx'
		)

		assert filtered_myelinated['gratio'].tolist() == [0.5]
		assert filtered_unmyelinated['axon_diam (um)'].tolist() == [6, 8]
		assert filtered_unmyelinated['solidity'].tolist() == [0.8, 0.95]
		assert filtered_unmyelinated['axon_area (um^2)'].tolist() == [10, 5]

	@pytest.mark.integration
	def test_main_overwrites_single_myelinated_file(self, tmp_path):
		myelinated_file = tmp_path / 'sample_axon_morphometrics.xlsx'
		config_file = tmp_path / 'filter.yaml'

		pd.DataFrame({
			'Unnamed: 0': [0, 1],
			'gratio': [0.5, 1.1],
			'axon_diam (um)': [5, 10],
		}).to_excel(myelinated_file, index=False)
		config_file.write_text(
			'myelinated:\n'
			'  - valid-g-ratio-only: true\n'
			'  - axon-diam-gt: null\n'
			'unmyelinated: []\n'
		)

		main([
			'--input', str(myelinated_file),
			'--config', str(config_file),
			'--overwrite',
		])

		assert not (tmp_path / 'sample_axon_morphometrics_filtered.xlsx').exists()
