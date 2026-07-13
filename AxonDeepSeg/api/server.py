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
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import imageio
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import get_file_extension, imread, imwrite
from AxonDeepSeg.params import axon_suffix, axonmyelin_suffix, myelin_suffix
from AxonDeepSeg.segment import (
    DEFAULT_MODEL_PATH,
    get_model_input_format,
    get_model_type,
)

MODEL_PATH = DEFAULT_MODEL_PATH

MASK_SUFFIXES = {
    'axon': axon_suffix,
    'myelin': myelin_suffix,
    'axonmyelin': axonmyelin_suffix,
}

# nnU-Net's predictor writes into the input's directory and mutates process-global
# state (os.environ, PIL's pixel limit), and spawns its own worker pool. Concurrent
# segmentations would race, so inference is serialized.
_INFERENCE_LOCK = threading.Lock()

_JOBS: Dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


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
    gpu_id: int = -1,
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

    try:
        # Inference blocks (torch); keep it off the event loop.
        masks = await asyncio.to_thread(run_segmentation, image_bytes, suffix)
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
)


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
    return {'model_available': True, 'model': MODEL_PATH.name}


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
