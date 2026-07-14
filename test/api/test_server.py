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
from skimage import draw

from AxonDeepSeg.api.server import app, resolve_gpu_id
from AxonDeepSeg.apply_model import clear_predictor_cache
from AxonDeepSeg.params import intensity


def _fake_masks(shape=(16, 16)):
    """Build a mask triplet using the ADS intensity encoding."""
    axon = np.zeros(shape, dtype=np.uint8)
    myelin = np.zeros(shape, dtype=np.uint8)
    axon[2:6, 2:6] = intensity['axon']
    myelin[8:12, 8:12] = intensity['myelin']
    axonmyelin = np.maximum(axon, myelin)

    return {'axon': axon, 'myelin': myelin, 'axonmyelin': axonmyelin}


def _fake_myelinated_axon(shape=(64, 64), centre=(32, 32), axon_r=8, myelin_r=13):
    """A single myelinated axon: a filled disk ringed by an annulus of myelin.

    Real morphometrics needs a real axon shape -- _fake_masks' squares have no
    myelin wrapped around them, so no g-ratio can be computed.
    """
    axon = np.zeros(shape, dtype=np.uint8)
    myelin = np.zeros(shape, dtype=np.uint8)

    rr, cc = draw.disk(centre, axon_r, shape=shape)
    axon[rr, cc] = intensity['axon']

    rr, cc = draw.disk(centre, myelin_r, shape=shape)
    myelin[rr, cc] = intensity['myelin']
    myelin[axon > 0] = 0  # annulus only

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
        clear_predictor_cache()
        if self.tmpDir.exists():
            shutil.rmtree(self.tmpDir)

    def _post(self, filename='image.png'):
        return self.client.post(
            '/segment',
            files={'file': (filename, self.uploadBytes, 'image/png')},
        )

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
        # The demo page fires this on every file-dialog open.
        with patch('AxonDeepSeg.api.server.get_predictor'):
            first = self.client.post('/warmup')
            second = self.client.post('/warmup')

        assert first.status_code == 200
        assert second.json()['warm'] is True

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
    def test_segment_returns_the_masks_directly(self):
        # Synchronous: inference is ~2s on a GPU, so there is no reason to make the
        # client poll a job. No job store also means no single-replica constraint.
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            response = self._post()

        assert response.status_code == 200
        body = response.json()
        assert body['meta']['shape'] == [16, 16]
        for mask_name in ['axon', 'myelin', 'axonmyelin']:
            assert body[mask_name]

    @pytest.mark.unit
    def test_returned_masks_are_valid_base64_png(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            body = self._post().json()

        for mask_name in ['axon', 'myelin', 'axonmyelin']:
            decoded = imageio.imread(base64.b64decode(body[mask_name]))
            assert decoded.shape == (16, 16)

    @pytest.mark.unit
    def test_returned_masks_use_ads_intensity_encoding(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_masks(),
        ):
            body = self._post().json()

        axonmyelin = imageio.imread(base64.b64decode(body['axonmyelin']))
        expected = {
            intensity['background'],
            intensity['myelin'],
            intensity['axon'],
        }

        assert set(np.unique(axonmyelin)).issubset(expected)

    # --------------failure handling tests-------------- #
    @pytest.mark.unit
    def test_inference_failure_returns_500_and_server_survives(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            side_effect=RuntimeError('boom'),
        ):
            response = self._post()

        assert response.status_code == 500
        # The process must still be serving: segment_images()'s sys.exit(2) would
        # have torn the worker down instead of failing this one request.
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
            response = self._post(filename='axon1.png')

        assert response.status_code == 200
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
                self._post()

        assert mock_run.call_args.kwargs['gpu_id'] == 0

    # --------------POST /morphometrics tests-------------- #
    def _post_morphometrics(self, pixel_size='0.07', axon_shape=None):
        data = {'pixel_size': pixel_size}
        if axon_shape is not None:
            data['axon_shape'] = axon_shape

        return self.client.post(
            '/morphometrics',
            files={'file': ('image.png', self.uploadBytes, 'image/png')},
            data=data,
        )

    @pytest.mark.unit
    def test_morphometrics_without_pixel_size_returns_422(self):
        # Diameters and g-ratios are meaningless without it, so it is not optional.
        response = self.client.post(
            '/morphometrics',
            files={'file': ('image.png', self.uploadBytes, 'image/png')},
        )

        assert response.status_code == 422

    @pytest.mark.unit
    def test_morphometrics_with_bad_axon_shape_returns_400(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_myelinated_axon(),
        ):
            response = self._post_morphometrics(axon_shape='hexagon')

        assert response.status_code == 400

    @pytest.mark.unit
    def test_morphometrics_returns_one_row_per_axon(self):
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_myelinated_axon(),
        ):
            response = self._post_morphometrics()

        assert response.status_code == 200
        body = response.json()
        assert body['meta']['n_axons'] == 1
        assert len(body['morphometrics']) == 1

        axon = body['morphometrics'][0]
        assert axon['gratio'] > 0
        assert axon['axon_diam (um)'] > 0

    @pytest.mark.unit
    def test_morphometrics_scales_with_pixel_size(self):
        # The pixel size is the only thing turning pixels into micrometres, so
        # doubling it must double the reported diameters.
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_myelinated_axon(),
        ):
            small = self._post_morphometrics(pixel_size='0.1').json()
            large = self._post_morphometrics(pixel_size='0.2').json()

        small_diam = small['morphometrics'][0]['axon_diam (um)']
        large_diam = large['morphometrics'][0]['axon_diam (um)']

        assert large_diam == pytest.approx(2 * small_diam, rel=1e-6)

    @pytest.mark.unit
    def test_morphometrics_also_returns_the_masks(self):
        # One inference, everything the caller needs: no reason to make them POST
        # the same image to /segment as well.
        with patch(
            'AxonDeepSeg.api.server.run_segmentation',
            return_value=_fake_myelinated_axon(),
        ):
            body = self._post_morphometrics().json()

        for mask_name in ['axon', 'myelin', 'axonmyelin']:
            assert imageio.imread(base64.b64decode(body[mask_name])).shape == (64, 64)

    # --------------end-to-end tests-------------- #
    @pytest.mark.integration
    def test_end_to_end_morphometrics_real_image(self):
        response = self.client.post(
            '/morphometrics',
            files={
                'file': (
                    'image.png',
                    self.demoImagePath.read_bytes(),
                    'image/png',
                )
            },
            data={'pixel_size': '0.07', 'axon_shape': 'circle'},
        )

        assert response.status_code == 200, response.text
        body = response.json()

        # The demo image is dense with axons; a real run must find many.
        assert body['meta']['n_axons'] > 10
        assert len(body['morphometrics']) == body['meta']['n_axons']

        first = body['morphometrics'][0]
        for column in ['x0 (px)', 'y0 (px)', 'gratio', 'axon_diam (um)']:
            assert column in first

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

        assert response.status_code == 200
        body = response.json()
        assert body['meta']['shape'] == expected_shape

        axon = imageio.imread(base64.b64decode(body['axon']))
        axonmyelin = imageio.imread(base64.b64decode(body['axonmyelin']))

        assert list(axon.shape) == expected_shape
        # A real segmentation of the demo image must actually find axons.
        assert np.any(axon == intensity['axon'])
        assert np.any(axonmyelin == intensity['myelin'])
