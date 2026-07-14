"""
HTTP service wrapping the AxonDeepSeg segmentation API.

POST /segment accepts an image, runs nnU-Net inference, and returns the axon and
myelin masks as base64-encoded PNGs. Work is done asynchronously: the POST returns
202 with a job id, and the client polls GET /segment/{job_id} for the result.

Deployment note: jobs are held in memory, so this serves correctly only as a single
replica. Behind a round-robin load balancer a client that POSTs to one pod and polls
another gets a 404. Multi-replica deployment needs a shared job store (e.g. Redis)
or masks persisted to object storage.
"""

import asyncio
import base64
import io
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import imageio
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from AxonDeepSeg.apply_model import (
    axon_segmentation,
    get_predictor,
    is_predictor_cached,
)
from AxonDeepSeg.ads_utils import get_file_extension, imread, imwrite
from AxonDeepSeg.params import axon_suffix, axonmyelin_suffix, myelin_suffix
from AxonDeepSeg.segment import (
    DEFAULT_MODEL_PATH,
    get_model_input_format,
    get_model_type,
)

MODEL_PATH = DEFAULT_MODEL_PATH

# Reference frontend, served same-origin so the browser needs no CORS config.
DEMO_PAGE_PATH = Path(__file__).parent / 'static' / 'index.html'

MASK_SUFFIXES = {
    'axon': axon_suffix,
    'myelin': myelin_suffix,
    'axonmyelin': axonmyelin_suffix,
}

# Inference is serialized, and this is REQUIRED FOR CORRECTNESS -- do not remove it to
# get concurrency. The predictor is now cached and shared across requests, and nnU-Net
# mutates its network in place (network.load_state_dict is called per image) while the
# forward pass releases the GIL. Two requests predicting at once would race and produce
# silently wrong segmentations. nnU-Net also mutates process-global state (os.environ,
# PIL's pixel limit) and writes into the input's directory.
_INFERENCE_LOCK = threading.Lock()

_JOBS: Dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


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


@dataclass
class Job:
    id: str
    status: str = 'pending'  # pending | running | done | failed
    masks: Optional[dict] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


def run_segmentation(
    image_bytes: bytes,
    suffix: str = '.png',
    model_path: Optional[Path] = None,
    gpu_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Segment a single in-memory image.

    Only the file *extension* of the upload is taken from the client, never its
    name: merge_masks() derives the merged mask's name via
    name.replace('axon', 'axonmyelin'), so an upload called 'axon1.png' would
    produce a mangled output. The image is staged under a neutral stem instead.

    Note this deliberately bypasses segment.segment_images(), which calls
    sys.exit(2) on bad input and is wrapped in @logger.catch (reraise=False), so it
    swallows failures and returns None either way. axon_segmentation() lets
    exceptions propagate, which is what a server needs.

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

    file_format, n_channels = get_model_input_format(model_path)
    if n_channels != 1:
        raise ValueError(
            f'Model expects {n_channels}-channel input; only grayscale is supported.'
        )

    with _INFERENCE_LOCK:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            upload_path = tmp / f'upload{suffix}'
            upload_path.write_bytes(image_bytes)

            # imread() grayscales and normalizes bitdepth; imwrite() puts the image
            # in the format the model's dataset.json asks for.
            image = imread(str(upload_path))
            input_path = tmp / f'input{file_format}'
            imwrite(str(input_path), image, file_format)

            axon_segmentation(
                path_inputs=[input_path],
                path_model=model_path,
                model_type=get_model_type(model_path),
                gpu_id=gpu_id,
            )

            masks = {}
            for name, mask_suffix in MASK_SUFFIXES.items():
                mask_path = tmp / (input_path.stem + str(mask_suffix))
                if not mask_path.exists():
                    raise RuntimeError(
                        f'Segmentation did not produce a {name} mask.'
                    )
                masks[name] = imread(str(mask_path))

    return masks


def _encode_png(array: np.ndarray) -> str:
    """PNG-encode a mask and base64 it for JSON transport."""
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format='png')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


async def _run_job(job_id: str, image_bytes: bytes, suffix: str) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].status = 'running'

    gpu_id = resolve_gpu_id()

    try:
        # Inference blocks (torch); keep it off the event loop.
        masks = await asyncio.to_thread(
            run_segmentation, image_bytes, suffix, gpu_id=gpu_id
        )
    except (Exception, SystemExit) as exc:
        # SystemExit is caught on purpose: several functions down the ADS call chain
        # exit the process on bad input, which must fail this job rather than take
        # the whole server down with it.
        logger.exception(f'Segmentation job {job_id} failed.')
        with _JOBS_LOCK:
            _JOBS[job_id].status = 'failed'
            _JOBS[job_id].error = f'{type(exc).__name__}: {exc}'
        return

    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job.masks = masks
        job.meta = {
            'model': MODEL_PATH.name,
            'shape': list(masks['axonmyelin'].shape[:2]),
        }
        job.status = 'done'


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


@app.post('/segment', status_code=202)
async def segment(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """Queue a segmentation. Returns a job id to poll."""
    suffix = get_file_extension(file.filename or '')
    if suffix is None or 'ome' in suffix:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.filename}'. Supported extensions: "
                "'.png', '.tif', '.tiff', '.jpg', '.jpeg'."
            ),
        )

    image_bytes = await file.read()

    job = Job(id=str(uuid.uuid4()))
    with _JOBS_LOCK:
        _JOBS[job.id] = job

    background_tasks.add_task(_run_job, job.id, image_bytes, suffix)

    return {'job_id': job.id, 'status': job.status}


@app.get('/segment/{job_id}')
def get_segmentation(job_id: str) -> dict:
    """Poll a segmentation job; returns the masks once it is done."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f'Unknown job {job_id}.')

    body = {'job_id': job.id, 'status': job.status}

    if job.status == 'failed':
        body['error'] = job.error
    elif job.status == 'done':
        body['meta'] = job.meta
        for name, mask in job.masks.items():
            body[name] = _encode_png(mask)

    return body


def main():
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
