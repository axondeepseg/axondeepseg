# coding: utf-8

import base64
import io
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import imageio
import numpy as np
import pytest
from fastapi.testclient import TestClient

from AxonDeepSeg.api.server import app, resolve_gpu_id, _JOBS
from AxonDeepSeg.apply_model import clear_predictor_cache
from AxonDeepSeg.params import intensity


def _png_bytes(array):
    """Encode a numpy array as PNG bytes."""
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format='png')
    return buffer.getvalue()


def _fake_masks(shape=(16, 16)):
    """Build a mask triplet using the ADS intensity encoding."""
    axon = np.zeros(shape, dtype=np.uint8)
    myelin = np.zeros(shape, dtype=np.uint8)
    axon[2:6, 2:6] = intensity['axon']
    myelin[8:12, 8:12] = intensity['myelin']
    axonmyelin = np.maximum(axon, myelin)

    return {'axon': axon, 'myelin': myelin, 'axonmyelin': axonmyelin}


class TestCore(object):
    def setup_method(self):
        self.testPath = Path(__file__).resolve().parent.parent
        self.projectPath = self.testPath.parent

        self.modelPath = (
            self.projectPath /
            'AxonDeepSeg' /
            'models' /
            'model_seg_generalist_light'
            )

        self.demoImagePath = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__' /
            'image.png'
            )

        self.client = TestClient(app)

        # A synthetic upload, so the unit tests don't depend on downloaded fixtures.
        self.tmpDir = Path(tempfile.mkdtemp())
        self.uploadPath = self.tmpDir / 'image.png'
        imageio.imwrite(
            self.uploadPath,
            np.full((16, 16), 128, dtype=np.uint8),
            format='png',
        )
        self.uploadBytes = self.uploadPath.read_bytes()

    def teardown_method(self):
        _JOBS.clear()
        clear_predictor_cache()
        if self.tmpDir.exists():
            shutil.rmtree(self.tmpDir)

    # --------------warmup tests-------------- #
    @pytest.mark.unit
    def test_warmup_loads_the_predictor_and_reports_warm(self):
        with patch('AxonDeepSeg.api.server.get_predictor') as mock_get:
            response = self.client.post('/warmup')

        assert response.status_code == 200
        body = response.json()
        assert body['warm'] is True
        assert 'load_seconds' in body
        mock_get.assert_called_once()

    @pytest.mark.unit
    def test_warmup_is_safe_to_call_repeatedly(self):
        # The front-end fires this on every file-dialog open, so it must be idempotent.
        with patch('AxonDeepSeg.api.server.get_predictor'):
            first = self.client.post('/warmup')
            second = self.client.post('/warmup')

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()['warm'] is True

    @pytest.mark.unit
    def test_ready_reports_a_cold_predictor_as_not_warm(self):
        with patch('AxonDeepSeg.api.server.MODEL_PATH', self.tmpDir):
            body = self.client.get('/ready').json()

        assert body['warm'] is False

    @pytest.mark.unit
    def test_ready_reports_a_loaded_predictor_as_warm(self):
        with patch('AxonDeepSeg.api.server.MODEL_PATH', self.tmpDir):
            with patch(
                'AxonDeepSeg.api.server.is_predictor_cached',
                return_value=True,
            ):
                body = self.client.get('/ready').json()

        assert body['warm'] is True

    # --------------health/readiness tests-------------- #
    @pytest.mark.unit
    def test_health_returns_ok(self):
        response = self.client.get('/health')

        assert response.status_code == 200
        assert response.json()['status'] == 'ok'

    @pytest.mark.unit
    def test_ready_returns_200_when_model_present(self):
        with patch('AxonDeepSeg.api.server.MODEL_PATH', self.tmpDir):
            response = self.client.get('/ready')

        assert response.status_code == 200
        assert response.json()['model_available'] is True

    @pytest.mark.unit
    def test_ready_returns_503_when_model_missing(self):
        with patch('AxonDeepSeg.api.server.MODEL_PATH', self.tmpDir / 'nope'):
            response = self.client.get('/ready')

        assert response.status_code == 503

    # --------------POST /segment tests-------------- #
    @pytest.mark.unit
    def test_segment_without_file_returns_422(self):
        response = self.client.post('/segment')

        assert response.status_code == 422

    @pytest.mark.unit
    def test_segment_with_unsupported_extension_returns_400(self):
        response = self.client.post(
            '/segment',
            files={'file': ('image.bmp', self.uploadBytes, 'image/bmp')},
        )

        assert response.status_code == 400

    @pytest.mark.unit
    def test_segment_returns_202_with_job_id(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            response = self.client.post(
                '/segment',
                files={'file': ('image.png', self.uploadBytes, 'image/png')},
            )

        assert response.status_code == 202
        body = response.json()
        assert body['job_id']
        assert body['status'] == 'pending'

    # --------------GET /segment/{job_id} tests-------------- #
    @pytest.mark.unit
    def test_get_unknown_job_returns_404(self):
        response = self.client.get('/segment/does-not-exist')

        assert response.status_code == 404

    @pytest.mark.unit
    def test_job_reaches_done_and_returns_three_masks(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            job_id = self.client.post(
                '/segment',
                files={'file': ('image.png', self.uploadBytes, 'image/png')},
            ).json()['job_id']

            response = self.client.get(f'/segment/{job_id}')

        assert response.status_code == 200
        body = response.json()
        assert body['status'] == 'done'
        assert body['meta']['shape'] == [16, 16]
        for mask_name in ['axon', 'myelin', 'axonmyelin']:
            assert body[mask_name]

    @pytest.mark.unit
    def test_returned_masks_are_valid_base64_png(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            job_id = self.client.post(
                '/segment',
                files={'file': ('image.png', self.uploadBytes, 'image/png')},
            ).json()['job_id']

            body = self.client.get(f'/segment/{job_id}').json()

        for mask_name in ['axon', 'myelin', 'axonmyelin']:
            decoded = imageio.imread(base64.b64decode(body[mask_name]))
            assert decoded.shape == (16, 16)

    @pytest.mark.unit
    def test_returned_masks_use_ads_intensity_encoding(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            job_id = self.client.post(
                '/segment',
                files={'file': ('image.png', self.uploadBytes, 'image/png')},
            ).json()['job_id']

            body = self.client.get(f'/segment/{job_id}').json()

        axonmyelin = imageio.imread(base64.b64decode(body['axonmyelin']))
        expected = {
            intensity['background'],
            intensity['myelin'],
            intensity['axon'],
        }

        assert set(np.unique(axonmyelin)).issubset(expected)

    # --------------failure handling tests-------------- #
    @pytest.mark.unit
    def test_inference_exception_marks_job_failed_and_server_survives(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            side_effect=RuntimeError('boom'),
        ):
            job_id = self.client.post(
                '/segment',
                files={'file': ('image.png', self.uploadBytes, 'image/png')},
            ).json()['job_id']

            body = self.client.get(f'/segment/{job_id}').json()

        assert body['status'] == 'failed'
        assert 'boom' in body['error']
        # The process must still be serving: segment_images()'s sys.exit(2) would
        # have torn the worker down instead of failing this one job.
        assert self.client.get('/health').status_code == 200

    @pytest.mark.unit
    def test_upload_filename_is_not_passed_to_inference(self):
        # merge_masks() renames via name.replace('axon', 'axonmyelin'), so a file
        # named 'axon1.png' would produce a mangled mask name. The server must
        # hand inference raw bytes, never the client's filename.
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ) as mock_run:
            job_id = self.client.post(
                '/segment',
                files={'file': ('axon1.png', self.uploadBytes, 'image/png')},
            ).json()['job_id']

            body = self.client.get(f'/segment/{job_id}').json()

        assert body['status'] == 'done'
        assert 'axon1' not in str(mock_run.call_args)

    # --------------gpu selection tests-------------- #
    @pytest.mark.unit
    def test_gpu_id_is_cpu_when_no_cuda_is_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('torch.cuda.is_available', return_value=False):
                assert resolve_gpu_id() == -1

    @pytest.mark.unit
    def test_gpu_id_defaults_to_first_gpu_when_cuda_is_available(self):
        # Without this, a container on a GPU instance would silently run on CPU.
        with patch.dict(os.environ, {}, clear=True):
            with patch('torch.cuda.is_available', return_value=True):
                assert resolve_gpu_id() == 0

    @pytest.mark.unit
    def test_gpu_id_env_var_overrides_autodetection(self):
        with patch.dict(os.environ, {'ADS_GPU_ID': '1'}, clear=True):
            with patch('torch.cuda.is_available', return_value=True):
                assert resolve_gpu_id() == 1

    @pytest.mark.unit
    def test_segmentation_runs_on_the_resolved_gpu(self):
        with patch.dict(os.environ, {'ADS_GPU_ID': '0'}, clear=True):
            with patch(
                'AxonDeepSeg.api.server.run_segmentation',
                return_value=_fake_masks(),
            ) as mock_run:
                self.client.post(
                    '/segment',
                    files={'file': ('image.png', self.uploadBytes, 'image/png')},
                )

        assert mock_run.call_args.kwargs['gpu_id'] == 0

    # --------------end-to-end test-------------- #
    @pytest.mark.integration
    def test_end_to_end_segment_real_image(self):
        expected_shape = list(imageio.imread(self.demoImagePath).shape[:2])

        response = self.client.post(
            '/segment',
            files={
                'file': (
                    'image.png',
                    self.demoImagePath.read_bytes(),
                    'image/png',
                )
            },
        )
        assert response.status_code == 202
        job_id = response.json()['job_id']

        body = self.client.get(f'/segment/{job_id}').json()
        assert body['status'] == 'done', body.get('error')
        assert body['meta']['shape'] == expected_shape

        axon = imageio.imread(base64.b64decode(body['axon']))
        axonmyelin = imageio.imread(base64.b64decode(body['axonmyelin']))

        assert list(axon.shape) == expected_shape
        # A real segmentation of the demo image must actually find axons.
        assert np.any(axon == intensity['axon'])
        assert np.any(axonmyelin == intensity['myelin'])
