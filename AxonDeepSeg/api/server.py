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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from AxonDeepSeg.apply_model import (
    get_predictor,
    is_predictor_cached,
    segment_image_array,
)
from AxonDeepSeg.ads_utils import get_file_extension, imread
from AxonDeepSeg.download_model import get_model_cards, get_model_dir_name
from AxonDeepSeg.morphometrics.compute_morphometrics import (
    get_axon_morphometrics,
    rearrange_column_names_for_saving,
)
from AxonDeepSeg.segment import (
    DEFAULT_MODEL_PATH,
    MODELS_PATH,
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
        One ndarray mask per class the model emits, using the ADS intensity encoding
        (foreground=255, background=0). An axon+myelin model additionally gets a merged
        'axonmyelin' mask (axon=255, myelin=127); a multi-class model (e.g. the 5-class
        unmyelinated one) returns its per-class masks -- {'axon', 'myelin', 'nuclei',
        'process', 'uaxon'} -- with no 'axonmyelin'.
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


def _pack_labels(label_img: np.ndarray) -> np.ndarray:
    """Pack a uint16/uint32 instance-label image into an 8-bit RGB image.

    Each axon+myelin fibre has a distinct integer label (1..N, 0 = background). A
    browser <canvas> truncates a 16-bit greyscale PNG to 8 bits, so we spread the
    label across the R/G/B channels instead -- lossless, canvas-native, and it lets
    the web client recover the exact id (id = R + G*256 + B*65536) to map a clicked
    pixel back to its morphometrics row (row = id - 1).
    """
    lab = label_img.astype(np.uint32)
    rgb = np.zeros((*lab.shape, 3), np.uint8)
    rgb[..., 0] = lab & 0xFF
    rgb[..., 1] = (lab >> 8) & 0xFF
    rgb[..., 2] = (lab >> 16) & 0xFF
    return rgb


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


async def _run_segmentation_safe(
    image_bytes: bytes,
    suffix: str,
    model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run inference off the event loop, turning any failure into a 500."""
    gpu_id = resolve_gpu_id()

    try:
        # Inference blocks (torch); keep it off the event loop.
        return await asyncio.to_thread(
            run_segmentation, image_bytes, suffix, model_path=model_path, gpu_id=gpu_id
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


async def _segment_upload(file: UploadFile, suffix: str) -> Dict[str, np.ndarray]:
    """Run inference on an upload, turning any failure into a 500."""
    image_bytes = await file.read()

    return await _run_segmentation_safe(image_bytes, suffix)


_MODEL_CARDS: Optional[dict] = None


def _model_cards() -> dict:
    """The parsed model registry, read from disk once (it never changes at runtime)."""
    global _MODEL_CARDS
    if _MODEL_CARDS is None:
        _MODEL_CARDS = get_model_cards()

    return _MODEL_CARDS


def _resolve_model_path(model_name: Optional[str]) -> Path:
    """
    Map a friendly model name (a model_cards.yaml key, e.g. 'dedicated-SEM') to its
    on-disk model directory.

    None means the default model. An unknown name is a client error (400); a known name
    whose weights are not present in this image is a server-side gap (503) -- the
    orchestrator baked in a different set of models.
    """
    if model_name is None:
        return MODEL_PATH

    try:
        dir_name = get_model_dir_name(model_name, _model_cards())
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'.",
        ) from None

    model_path = MODELS_PATH / dir_name
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is not available on this server.",
        )

    return model_path


def _masks_payload(masks: Dict[str, np.ndarray], model_name: Optional[str] = None) -> dict:
    """The mask half of a response: base64 PNGs plus the image's shape."""
    # 'axonmyelin' only exists for axon+myelin models; a multi-class model (e.g. the
    # 5-class unmyelinated one) returns per-class masks without it. Read the shape from any
    # mask so those models still segment cleanly instead of KeyError-ing into a 500.
    reference = masks['axonmyelin'] if 'axonmyelin' in masks else next(iter(masks.values()))
    body = {
        'meta': {
            'model': model_name or MODEL_PATH.name,
            'shape': list(reference.shape[:2]),
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


def _validate_axon_shape(axon_shape: str) -> None:
    """Reject an unsupported axon shape (before spending a segmentation on it)."""
    if axon_shape not in AXON_SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"axon_shape must be one of {AXON_SHAPES}, got '{axon_shape}'.",
        )


async def _morphometrics_payload(
    masks: Dict[str, np.ndarray],
    pixel_size: float,
    axon_shape: str,
    model_name: Optional[str] = None,
) -> dict:
    """Measure every myelinated axon in a set of masks and fold it into a masks payload.

    This is the myelinated-axon path: get_axon_morphometrics on the axon+myelin masks,
    the same computation the CLI's default mode runs. A model that emits extra classes
    alongside axon+myelin (e.g. the 5-class unmyelinated model, which also has uaxon,
    nuclei, process) is measured on its axon+myelin channels only -- exactly what the CLI
    does from that model's per-class mask files; the other classes have no morphometrics
    handling anywhere. Unmyelinated (uaxon) morphometrics is a separate mode (CLI's -u,
    get_axon_morphometrics with im_myelin=None) not yet wired into /invocations.
    """
    # Without both an axon and a myelin mask there is nothing to measure here; fail
    # clearly rather than KeyError below.
    if 'axon' not in masks or 'myelin' not in masks:
        raise HTTPException(
            status_code=400,
            detail='Morphometrics requires a model that produces axon and myelin masks.',
        )

    try:
        # return_im_axonmyelin_label gives the instance map: a labelled image where each
        # axon+myelin fibre has value i+1 for morphometrics row i. The call now returns a
        # tuple (stats_dataframe, im_axonmyelin_label).
        stats, im_axonmyelin_label = await asyncio.to_thread(
            get_axon_morphometrics,
            masks['axon'],
            None,
            masks['myelin'],
            pixel_size,
            axon_shape,
            return_im_axonmyelin_label=True,
        )
    except (Exception, SystemExit) as exc:
        logger.exception('Morphometrics failed.')
        raise HTTPException(
            status_code=500,
            detail=f'{type(exc).__name__}: {exc}',
        ) from None

    # Same column order and unit-bearing display names the xlsx/csv files use.
    stats = rearrange_column_names_for_saving(stats)

    body = _masks_payload(masks, model_name=model_name)
    body['meta'].update({
        'pixel_size': pixel_size,
        'axon_shape': axon_shape,
        'n_axons': len(stats),
    })
    # to_json handles NaN (-> null) and numpy scalars, which json.dumps chokes on.
    body['morphometrics'] = json.loads(stats.to_json(orient='records'))
    # Instance map for click<->row selection: RGB-packed labels, id = row + 1.
    body['instance_map'] = _encode_png(_pack_labels(im_axonmyelin_label))

    return body


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
    _validate_axon_shape(axon_shape)

    suffix = _validate_upload(file.filename)
    masks = await _segment_upload(file, suffix)

    return await _morphometrics_payload(masks, pixel_size, axon_shape)


# --- SageMaker container contract -------------------------------------------------
# A SageMaker endpoint invokes the container on two routes it owns: GET /ping for health
# and POST /invocations for inference. For an *async* endpoint SageMaker does the S3
# download/upload itself -- it POSTs the input bytes here and writes our response body
# back to S3 -- so this is purely request-in/response-out; the container never touches S3.
# These are additional surface; /segment and /morphometrics stay for direct/local use.


@app.get('/ping')
def ping():
    """SageMaker health check: 200 once this replica can serve, 503 otherwise."""
    if not MODEL_PATH.exists():
        return JSONResponse(status_code=503, content={'status': 'unavailable'})

    return {'status': 'ok'}


@app.post('/invocations')
async def invocations(request: Request) -> dict:
    """
    SageMaker inference entry point.

    Body is a JSON envelope:
        {
          "mode": "segment" | "morphometrics",   # default "segment"
          "model": "generalist",                 # optional; default model if omitted
          "image": "<base64-encoded image>",     # required
          "pixel_size": 0.07,                     # required for morphometrics
          "axon_shape": "circle",                # optional; default "circle"
          "filename": "sample.tif"               # optional; only its extension is used
        }

    Returns the same JSON shape as /segment (or /morphometrics).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='Body must be valid JSON.') from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail='Body must be a JSON object.')

    mode = body.get('mode', 'segment')
    if mode not in ('segment', 'morphometrics'):
        raise HTTPException(
            status_code=400,
            detail=f"mode must be 'segment' or 'morphometrics', got '{mode}'.",
        )

    image_b64 = body.get('image')
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing required field 'image'.")
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400, detail="'image' is not valid base64."
        ) from None

    # Validate the cheap, request-shaped things before spending a segmentation.
    axon_shape = body.get('axon_shape', 'circle')
    pixel_size = None
    if mode == 'morphometrics':
        _validate_axon_shape(axon_shape)
        if body.get('pixel_size') is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required field 'pixel_size' for morphometrics.",
            )
        try:
            pixel_size = float(body['pixel_size'])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"pixel_size must be a number, got '{body['pixel_size']}'.",
            ) from None

    suffix = _validate_upload(body.get('filename') or 'upload.png')
    model_name = body.get('model')
    model_path = _resolve_model_path(model_name)

    masks = await _run_segmentation_safe(image_bytes, suffix, model_path=model_path)

    # Echo the friendly name the client sent so it can match the response to its request
    # (falling back to the directory name for the default model). A compare-all caller
    # running several models labels each result by this.
    model_label = model_name or model_path.name

    if mode == 'segment':
        return _masks_payload(masks, model_name=model_label)

    return await _morphometrics_payload(
        masks, pixel_size, axon_shape, model_name=model_label
    )


def _bind_port() -> int:
    """
    Port to bind. SageMaker sets SAGEMAKER_BIND_TO_PORT (its container serves on 8080);
    ADS_PORT is the local override. Default 8000 keeps existing direct/local use.
    """
    return int(os.environ.get('SAGEMAKER_BIND_TO_PORT') or os.environ.get('ADS_PORT') or 8000)


def main():
    import uvicorn

    # main() ignores argv, so SageMaker's `serve` argument is harmless -- but only because
    # the Dockerfile uses ENTRYPOINT ["ads_server"], which makes `docker run <image> serve`
    # resolve to `ads_server serve` rather than trying to exec a `serve` binary.
    uvicorn.run(app, host='0.0.0.0', port=_bind_port())


if __name__ == '__main__':
    main()
