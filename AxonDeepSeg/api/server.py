"""
HTTP service wrapping the AxonDeepSeg segmentation API.

POST /segment accepts an image, runs nnU-Net inference, and returns the axon and
myelin masks as base64-encoded PNGs in the response.

It is synchronous. An earlier version returned a job id for the client to poll,
because a segmentation took ~50s. On a GPU, with the predictor cached and nnU-Net's
per-call worker pool bypassed, it takes ~1.8s -- so there is nothing to poll for, and
holding no job state means the service scales horizontally: any replica can answer any
request.

Inference is serialized within a process (see _INFERENCE_LOCK), so concurrent requests
queue rather than run in parallel. To serve more concurrency, run more replicas.
"""

import asyncio
import base64
import io
import json
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import imageio
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from AxonDeepSeg.apply_model import (
    get_predictor,
    is_predictor_cached,
    segment_image_array,
)
from AxonDeepSeg.ads_utils import get_file_extension, imread
from AxonDeepSeg.morphometrics.compute_morphometrics import (
    get_axon_morphometrics,
    rearrange_column_names_for_saving,
)
from AxonDeepSeg.segment import (
    DEFAULT_MODEL_PATH,
    get_model_input_format,
    get_model_type,
)

AXON_SHAPES = ('circle', 'ellipse')

MODEL_PATH = DEFAULT_MODEL_PATH

# Reference frontend, served same-origin so the browser needs no CORS config.
DEMO_PAGE_PATH = Path(__file__).parent / 'static' / 'index.html'

# Inference is serialized, and this is REQUIRED FOR CORRECTNESS -- do not remove it to
# get concurrency. The predictor is now cached and shared across requests, and nnU-Net
# mutates its network in place (network.load_state_dict is called per image) while the
# forward pass releases the GIL. Two requests predicting at once would race and produce
# silently wrong segmentations. nnU-Net also mutates process-global state (os.environ,
# PIL's pixel limit) and writes into the input's directory.
_INFERENCE_LOCK = threading.Lock()


def resolve_gpu_id() -> int:
    """
    Which GPU to run inference on; -1 means CPU.

    Defaults to the first GPU whenever CUDA is available, so that a container
    scheduled onto a GPU instance actually uses it — the ADS default is -1 (CPU),
    which would otherwise mean paying for a GPU and never touching it.
    Set ADS_GPU_ID to pin a specific device, or to -1 to force CPU.
    """
    configured = os.environ.get('ADS_GPU_ID')
    if configured is not None:
        return int(configured)

    return 0 if torch.cuda.is_available() else -1


def warm_up() -> float:
    """
    Load the model into memory. Blocking, idempotent, returns seconds spent.

    A container that has not done this pays the checkpoint load on the user's first
    request. Calling it while the user is still picking a file in their browser hides
    that cost behind their own browse time.
    """
    started = time.perf_counter()
    get_predictor(MODEL_PATH, get_model_type(MODEL_PATH), resolve_gpu_id())

    return time.perf_counter() - started


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading the model as soon as the container boots, rather than waiting for
    # a request. On a thread, so /health answers immediately while it loads.
    if MODEL_PATH.exists():
        asyncio.create_task(asyncio.to_thread(warm_up))
    yield


def run_segmentation(
    image_bytes: bytes,
    suffix: str = '.png',
    model_path: Optional[Path] = None,
    gpu_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Segment a single in-memory image.

    Uses apply_model.segment_image_array(), not axon_segmentation(): the file-based
    path spawns ~8 worker processes per call, which measured as ~7.2s of an ~8.9s GPU
    request against ~1.7s of actual inference. Nothing here touches disk except
    decoding the upload.

    Also deliberately avoids segment.segment_images(), which calls sys.exit(2) on bad
    input and is wrapped in @logger.catch (reraise=False), so it swallows failures and
    returns None either way.

    Returns
    -------
    dict
        {'axon': ndarray, 'myelin': ndarray, 'axonmyelin': ndarray}, using the ADS
        intensity encoding (axon=255, myelin=127, background=0).
    """
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}.')

    if gpu_id is None:
        gpu_id = resolve_gpu_id()

    _, n_channels = get_model_input_format(model_path)
    if n_channels != 1:
        raise ValueError(
            f'Model expects {n_channels}-channel input; only grayscale is supported.'
        )

    # ads_utils.imread() needs a filename (it validates the extension and picks a
    # plugin), so the upload is staged briefly. Only the *extension* comes from the
    # client, never the filename.
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = Path(tmpdir) / f'upload{suffix}'
        upload_path.write_bytes(image_bytes)
        image = imread(str(upload_path))

    with _INFERENCE_LOCK:
        return segment_image_array(
            image,
            model_path,
            model_type=get_model_type(model_path),
            gpu_id=gpu_id,
        )


def _encode_png(array: np.ndarray) -> str:
    """PNG-encode a mask and base64 it for JSON transport."""
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format='png')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _validate_upload(filename: Optional[str]) -> str:
    """Return the upload's extension, or reject it."""
    suffix = get_file_extension(filename or '')
    if suffix is None or 'ome' in suffix:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{filename}'. Supported extensions: "
                "'.png', '.tif', '.tiff', '.jpg', '.jpeg'."
            ),
        )

    return suffix


async def _segment_upload(file: UploadFile, suffix: str) -> Dict[str, np.ndarray]:
    """Run inference on an upload, turning any failure into a 500."""
    image_bytes = await file.read()
    gpu_id = resolve_gpu_id()

    try:
        # Inference blocks (torch); keep it off the event loop.
        return await asyncio.to_thread(
            run_segmentation, image_bytes, suffix, gpu_id=gpu_id
        )
    except (Exception, SystemExit) as exc:
        # SystemExit is caught on purpose: several functions down the ADS call chain
        # exit the process on bad input, which must fail this request rather than take
        # the whole server down with it.
        logger.exception('Segmentation failed.')
        raise HTTPException(
            status_code=500,
            detail=f'{type(exc).__name__}: {exc}',
        ) from None


def _masks_payload(masks: Dict[str, np.ndarray]) -> dict:
    """The mask half of a response: base64 PNGs plus the image's shape."""
    body = {
        'meta': {
            'model': MODEL_PATH.name,
            'shape': list(masks['axonmyelin'].shape[:2]),
        }
    }
    for name, mask in masks.items():
        body[name] = _encode_png(mask)

    return body


app = FastAPI(
    title='AxonDeepSeg',
    description='Axon and myelin segmentation from microscopy images.',
    lifespan=lifespan,
)


@app.get('/', include_in_schema=False)
def demo_page():
    """A minimal browser client for the segmentation API."""
    return FileResponse(DEMO_PAGE_PATH, media_type='text/html')


@app.get('/health')
def health() -> dict:
    """Liveness: the process is up."""
    return {'status': 'ok'}


@app.get('/ready')
def ready():
    """Readiness: model weights are on disk, so this replica can actually serve."""
    available = MODEL_PATH.exists()
    if not available:
        return JSONResponse(
            status_code=503,
            content={'model_available': False, 'model': str(MODEL_PATH)},
        )

    gpu_id = resolve_gpu_id()

    return {
        'model_available': True,
        'model': MODEL_PATH.name,
        # Surfaced so a GPU deployment can be spotted running on CPU by mistake.
        'device': 'cpu' if gpu_id < 0 else f'cuda:{gpu_id}',
        # False means the next segmentation pays the checkpoint load first.
        'warm': is_predictor_cached(MODEL_PATH, gpu_id),
    }


@app.post('/warmup')
async def warmup() -> dict:
    """
    Load the model, so the next segmentation doesn't have to.

    Idempotent and cheap once warm. The demo page fires this when the file dialog
    opens, which spends the user's browse time on the checkpoint load instead of
    making them wait for it after they hit Segment.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f'Model not found at {MODEL_PATH}.',
        )

    gpu_id = resolve_gpu_id()
    load_seconds = await asyncio.to_thread(warm_up)

    return {
        'warm': True,
        'device': 'cpu' if gpu_id < 0 else f'cuda:{gpu_id}',
        'load_seconds': round(load_seconds, 2),
    }


@app.post('/segment')
async def segment(file: UploadFile = File(...)) -> dict:
    """Segment an image and return the masks as base64-encoded PNGs."""
    suffix = _validate_upload(file.filename)
    masks = await _segment_upload(file, suffix)

    return _masks_payload(masks)


@app.post('/morphometrics')
async def morphometrics(
    file: UploadFile = File(...),
    pixel_size: float = Form(..., description='Pixel size in micrometres.'),
    axon_shape: str = Form('circle'),
) -> dict:
    """
    Segment an image and measure every axon in it.

    Returns one row per axon (diameter, g-ratio, myelin thickness, ...) alongside the
    masks, so a caller that wants both does not have to pay for inference twice.

    pixel_size is required and has no default: it is the only thing turning pixels into
    micrometres, and a wrong one silently scales every diameter and thickness in the
    result. The CLI reads it from a pixel_size_in_micrometer.txt beside the image;
    over HTTP there is no such file, so the caller must say.
    """
    if axon_shape not in AXON_SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"axon_shape must be one of {AXON_SHAPES}, got '{axon_shape}'.",
        )

    suffix = _validate_upload(file.filename)
    masks = await _segment_upload(file, suffix)

    try:
        stats = await asyncio.to_thread(
            get_axon_morphometrics,
            masks['axon'],
            None,
            masks['myelin'],
            pixel_size,
            axon_shape,
        )
    except (Exception, SystemExit) as exc:
        logger.exception('Morphometrics failed.')
        raise HTTPException(
            status_code=500,
            detail=f'{type(exc).__name__}: {exc}',
        ) from None

    # Same column order and unit-bearing display names the xlsx/csv files use.
    stats = rearrange_column_names_for_saving(stats)

    body = _masks_payload(masks)
    body['meta'].update({
        'pixel_size': pixel_size,
        'axon_shape': axon_shape,
        'n_axons': len(stats),
    })
    # to_json handles NaN (-> null) and numpy scalars, which json.dumps chokes on.
    body['morphometrics'] = json.loads(stats.to_json(orient='records'))

    return body


def main():
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
