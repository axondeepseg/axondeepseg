from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from AxonDeepSeg.morphometrics.filter_morphometrics import (
	apply_myelinated_rules,
	apply_qa_flags,
	apply_unmyelinated_rules,
	main,
	read_config,
	read_qa_flags,
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


class TestQAFlags(object):
	@staticmethod
	def _morphometrics():
		return pd.DataFrame({
			'Unnamed: 0': [0, 1, 2, 3, 4],
			'gratio': [0.70, 0.75, 0.80, np.nan, 0.65],
			'axon_diam (um)': [1.0, 1.5, 2.0, 0.8, 1.2],
		})

	@staticmethod
	def _write_flags(path, rows):
		header = 'Axon index,Flag,Diameter (um),Myelin thickness (um),g-ratio\n'
		path.write_text(header + ''.join(f'{r}\n' for r in rows), encoding='utf-8')
		return path

	# --------------read_qa_flags tests-------------- #
	@pytest.mark.unit
	def test_read_qa_flags_reads_export(self, tmp_path):
		flags = read_qa_flags(self._write_flags(
			tmp_path / 'flags.csv', ['0,good,1.0,0.2,0.7000', '2,delete,2.0,0.3,0.8000']
		))

		assert flags['Axon index'].tolist() == [0, 2]
		assert flags['Flag'].tolist() == ['good', 'delete']

	@pytest.mark.unit
	def test_read_qa_flags_rejects_old_report_exports(self, tmp_path):
		old = tmp_path / 'old.csv'
		old.write_text(
			'Axon ID,Flag Status,Diameter (µm),g-ratio\n1,Good Segmentation,1.00,0.70\n',
			encoding='utf-8',
		)

		with pytest.raises(ValueError, match='older QA reports'):
			read_qa_flags(old)

	@pytest.mark.unit
	def test_read_qa_flags_rejects_unknown_flag(self, tmp_path):
		with pytest.raises(ValueError, match='Unknown flag'):
			read_qa_flags(self._write_flags(tmp_path / 'flags.csv', ['0,maybe,1.0,0.2,0.7']))

	@pytest.mark.unit
	def test_read_qa_flags_rejects_duplicates(self, tmp_path):
		with pytest.raises(ValueError, match='more than once'):
			read_qa_flags(self._write_flags(
				tmp_path / 'flags.csv', ['0,good,1.0,0.2,0.7', '0,delete,1.0,0.2,0.7']
			))

	# --------------apply_qa_flags tests-------------- #
	@pytest.mark.unit
	def test_apply_qa_flags_not_deleted_drops_only_deleted(self, tmp_path):
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', [
			'0,good,1.0,0.2,0.7000',
			'1,needs_correction,1.5,0.2,0.7500',
			'3,delete,0.8,0.1,NaN',
		]))

		result = apply_qa_flags(self._morphometrics(), flags, keep='not-deleted')

		assert result['Unnamed: 0'].tolist() == [0, 1, 2, 4]

	@pytest.mark.unit
	def test_apply_qa_flags_good_only_keeps_only_good(self, tmp_path):
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', [
			'0,good,1.0,0.2,0.7000',
			'4,good,1.2,0.3,0.6500',
			'2,delete,2.0,0.3,0.8000',
		]))

		result = apply_qa_flags(self._morphometrics(), flags, keep='good-only')

		assert result['Unnamed: 0'].tolist() == [0, 4]

	@pytest.mark.unit
	def test_apply_qa_flags_uses_axon_index_not_row_position(self, tmp_path):
		# morphometrics already filtered once: row positions no longer match axon indexes
		df = self._morphometrics().iloc[[0, 2, 4]].reset_index(drop=True)
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', ['4,delete,1.2,0.3,0.6500']))

		result = apply_qa_flags(df, flags)

		assert result['Unnamed: 0'].tolist() == [0, 2]

	@pytest.mark.unit
	def test_apply_qa_flags_raises_when_gratio_does_not_match(self, tmp_path):
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', ['1,delete,1.5,0.2,0.9000']))

		with pytest.raises(ValueError, match='does not match'):
			apply_qa_flags(self._morphometrics(), flags)

	@pytest.mark.unit
	def test_apply_qa_flags_raises_for_unknown_axon(self, tmp_path):
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', ['9,delete,1.5,0.2,0.75']))

		with pytest.raises(ValueError, match='not in the morphometrics file'):
			apply_qa_flags(self._morphometrics(), flags)

	@pytest.mark.unit
	def test_apply_qa_flags_rejects_unknown_keep_option(self, tmp_path):
		flags = read_qa_flags(self._write_flags(tmp_path / 'flags.csv', ['0,good,1.0,0.2,0.7']))

		with pytest.raises(ValueError):
			apply_qa_flags(self._morphometrics(), flags, keep='everything')

	# --------------main integration tests-------------- #
	@pytest.mark.integration
	def test_main_applies_qa_flags_to_xlsx(self, tmp_path):
		morpho_file = tmp_path / 'sample_axon_morphometrics.xlsx'
		self._morphometrics().to_excel(morpho_file, index=False)
		flags = self._write_flags(tmp_path / 'flags.csv', [
			'1,delete,1.5,0.2,0.7500', '4,needs_correction,1.2,0.3,0.6500'
		])
		config_file = tmp_path / 'filter.yaml'
		config_file.write_text('myelinated: []\nunmyelinated: []\n')

		main([
			'--input', str(morpho_file),
			'--config', str(config_file),
			'--qa-flags', str(flags),
		])

		result = pd.read_excel(tmp_path / 'sample_axon_morphometrics_filtered.xlsx')
		assert result.iloc[:, 0].tolist() == [0, 2, 3, 4]

	@pytest.mark.integration
	def test_main_applies_qa_flags_to_napari_csv(self, tmp_path):
		morpho_file = tmp_path / 'Morphometrics.csv'
		self._morphometrics().rename(columns={'Unnamed: 0': ''}).to_csv(morpho_file, index=False)
		flags = self._write_flags(tmp_path / 'flags.csv', [
			'0,good,1.0,0.2,0.7000', '2,good,2.0,0.3,0.8000', '1,delete,1.5,0.2,0.7500'
		])
		config_file = tmp_path / 'filter.yaml'
		config_file.write_text('myelinated: []\nunmyelinated: []\n')

		main([
			'--input', str(morpho_file),
			'--config', str(config_file),
			'--qa-flags', str(flags),
			'--qa-keep', 'good-only',
		])

		result = pd.read_csv(tmp_path / 'Morphometrics_filtered.csv')
		assert result.iloc[:, 0].tolist() == [0, 2]
		# the file must not be filtered a second time as an "unmyelinated" file
		assert not (tmp_path / 'Morphometrics_filtered_filtered.csv').exists()

	@pytest.mark.integration
	def test_main_rejects_qa_flags_for_a_folder(self, tmp_path):
		flags = self._write_flags(tmp_path / 'flags.csv', ['0,good,1.0,0.2,0.7'])

		with pytest.raises(SystemExit):
			main(['--input', str(tmp_path), '--qa-flags', str(flags)])
