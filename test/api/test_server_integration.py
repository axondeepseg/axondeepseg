# coding: utf-8

"""
End-to-end tests against a *real* uvicorn process.

test_server.py drives the app in-process with FastAPI's TestClient, which never
opens a socket and runs background tasks to completion before the response is
returned. That hides the behaviour that matters most here: whether POST /segment
really does return 202 immediately and finish the work afterwards. These tests
launch the server as a subprocess and talk to it over HTTP, and drive the demo
page in a real browser.
"""

import base64
import io
import socket
import subprocess
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import pytest
import requests

from AxonDeepSeg.params import intensity

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - playwright is an optional extra
    sync_playwright = None

# Segmenting the demo image takes ~30s on CPU; give it generous headroom for CI.
SEGMENTATION_TIMEOUT_S = 600
SERVER_BOOT_TIMEOUT_S = 120

_CHROMIUM_AVAILABLE = None


def _chromium_available():
    """Whether a chromium binary is installed (`playwright install chromium`)."""
    global _CHROMIUM_AVAILABLE

    if _CHROMIUM_AVAILABLE is None:
        if sync_playwright is None:
            _CHROMIUM_AVAILABLE = False
        else:
            try:
                with sync_playwright() as playwright:
                    playwright.chromium.launch().close()
                _CHROMIUM_AVAILABLE = True
            except Exception:
                _CHROMIUM_AVAILABLE = False

    return _CHROMIUM_AVAILABLE


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _decode_mask(encoded):
    return imageio.imread(io.BytesIO(base64.b64decode(encoded)))


class TestCore(object):
    def setup_method(self):
        self.testPath = Path(__file__).resolve().parent.parent
        self.projectPath = self.testPath.parent

        self.imagePath = (
            self.testPath /
            '__test_files__' /
            '__test_demo_files__' /
            'image.png'
            )

        self.port = _free_port()
        self.baseUrl = f'http://127.0.0.1:{self.port}'

        self.server = subprocess.Popen(
            [
                sys.executable, '-m', 'uvicorn',
                'AxonDeepSeg.api.server:app',
                '--host', '127.0.0.1',
                '--port', str(self.port),
            ],
            cwd=str(self.projectPath),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_server()

    def teardown_method(self):
        self.server.terminate()
        try:
            self.server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=30)

    def _wait_for_server(self):
        deadline = time.time() + SERVER_BOOT_TIMEOUT_S
        while time.time() < deadline:
            if self.server.poll() is not None:
                output = self.server.stdout.read().decode(errors='replace')
                raise RuntimeError(f'Server died on startup:\n{output}')
            try:
                if requests.get(f'{self.baseUrl}/health', timeout=2).ok:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)

        raise RuntimeError('Server did not become healthy in time.')

    def _poll_until_finished(self, job_id):
        deadline = time.time() + SEGMENTATION_TIMEOUT_S
        while time.time() < deadline:
            body = requests.get(f'{self.baseUrl}/segment/{job_id}', timeout=10).json()
            if body['status'] in ('done', 'failed'):
                return body
            time.sleep(1)

        raise AssertionError(f'Job {job_id} did not finish in time.')

    # --------------live HTTP tests-------------- #
    @pytest.mark.integration
    def test_live_server_is_ready(self):
        response = requests.get(f'{self.baseUrl}/ready', timeout=10)

        assert response.status_code == 200
        assert response.json()['model_available'] is True

    @pytest.mark.integration
    def test_post_returns_202_without_waiting_for_inference(self):
        # The whole point of the job API: the POST must come back immediately,
        # long before the ~30s segmentation has finished.
        start = time.time()
        with open(self.imagePath, 'rb') as image:
            response = requests.post(
                f'{self.baseUrl}/segment',
                files={'file': ('image.png', image, 'image/png')},
                timeout=30,
            )
        elapsed = time.time() - start

        assert response.status_code == 202
        assert response.json()['status'] == 'pending'
        assert elapsed < 10

        # And the job does finish afterwards.
        assert self._poll_until_finished(response.json()['job_id'])['status'] == 'done'

    @pytest.mark.integration
    def test_live_server_segments_over_http(self):
        expected_shape = list(imageio.imread(self.imagePath).shape[:2])

        with open(self.imagePath, 'rb') as image:
            job_id = requests.post(
                f'{self.baseUrl}/segment',
                files={'file': ('image.png', image, 'image/png')},
                timeout=30,
            ).json()['job_id']

        body = self._poll_until_finished(job_id)

        assert body['status'] == 'done', body.get('error')
        assert body['meta']['shape'] == expected_shape

        axon = _decode_mask(body['axon'])
        axonmyelin = _decode_mask(body['axonmyelin'])

        assert list(axon.shape) == expected_shape
        assert np.any(axon == intensity['axon'])
        assert np.any(axonmyelin == intensity['myelin'])

    @pytest.mark.integration
    def test_live_server_survives_a_failed_job(self):
        response = requests.post(
            f'{self.baseUrl}/segment',
            files={'file': ('image.png', b'not a real png', 'image/png')},
            timeout=30,
        )

        body = self._poll_until_finished(response.json()['job_id'])

        assert body['status'] == 'failed'
        assert body['error']
        # A crashed worker would fail this: the process must still be serving.
        assert requests.get(f'{self.baseUrl}/health', timeout=10).ok

    # --------------browser tests-------------- #
    @pytest.mark.integration
    def test_demo_page_is_served_at_root(self):
        response = requests.get(self.baseUrl, timeout=10)

        assert response.status_code == 200
        assert 'text/html' in response.headers['content-type']

    @pytest.mark.integration
    def test_demo_page_segments_image_in_browser(self):
        if not _chromium_available():
            pytest.skip("Chromium missing; run 'playwright install chromium'.")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            try:
                page.goto(self.baseUrl)

                # Drive the page the way a user would: pick a file, hit segment.
                page.set_input_files('#file-input', str(self.imagePath))
                page.click('#segment-button')

                # The page's own JS does the POST and polls the job endpoint.
                page.wait_for_selector(
                    '#status[data-status="done"]',
                    timeout=SEGMENTATION_TIMEOUT_S * 1000,
                )

                # The browser must have actually decoded and rendered the masks.
                for mask_name in ['axon', 'myelin', 'axonmyelin']:
                    source = page.get_attribute(f'#{mask_name}', 'src')
                    assert source.startswith('data:image/png;base64,')

                    width = page.evaluate(
                        f'document.getElementById("{mask_name}").naturalWidth'
                    )
                    assert width > 0
            finally:
                browser.close()
