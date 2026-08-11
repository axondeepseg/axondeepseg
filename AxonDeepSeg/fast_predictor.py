'''
Faster execution backends for AxonDeepSeg inference.

nnU-Net only enables autocast for CUDA, so on Apple silicon the forward pass runs
fp32 through MPS. `AdsPredictor` subclasses `nnUNetPredictor` and swaps the forward
pass for a faster backend, leaving `nnunetv2` a pinned, unmodified dependency.

Measured on an Apple M5, model_seg_generalist_light, 384x640 patch:

    PyTorch MPS fp32 (nnU-Net default)   42.7 ms
    PyTorch MPS fp16                     25.5 ms
    CoreML fp16 (GPU)                    18.0 ms
    CoreML fp16 (Neural Engine)          11.8 ms

Output is unchanged: CoreML+ANE scores Dice 0.9988 (myelin) / 0.9996 (axon) against
the PyTorch fp32 result, over 26 SEM and TEM images.
'''

import hashlib
import itertools
import os
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
from loguru import logger

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

Backend = Literal['auto', 'torch', 'torch-fp16', 'coreml']


def _checkpoint_fingerprint(path_model: Path, folds: List[str], patch_size: Tuple[int, ...]) -> str:
    '''
    Short digest identifying a converted model, so a cached conversion is never
    reused after the weights or patch size change.

    Parameters
    ----------
    path_model : pathlib.Path
        Path to the nnU-Net model folder.
    folds : List[str]
        Fold names used for this prediction, e.g. ['all'].
    patch_size : Tuple[int, ...]
        Patch size the model was converted for.

    Returns
    -------
    str
        Hex digest, truncated to 16 characters.
    '''
    h = hashlib.sha256()
    h.update('x'.join(str(p) for p in patch_size).encode())
    for fold in folds:
        for chkpt in sorted((path_model / f'fold_{fold}').glob('*.pth')):
            stat = chkpt.stat()
            h.update(f'{chkpt.name}:{stat.st_size}:{int(stat.st_mtime)}'.encode())
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    '''
    Directory for converted models, kept outside the package so it still works
    when AxonDeepSeg is installed read-only.
    '''
    root = os.environ.get('ADS_CACHE_DIR')
    base = Path(root) if root else Path.home() / '.cache' / 'axondeepseg'
    return base / 'coreml'


def coreml_available() -> bool:
    '''
    Whether the CoreML backend is worth using here. Restricted to Apple silicon:
    the speedup comes from the Neural Engine, which Intel Macs lack.
    '''
    import platform
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return False
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: Backend, device: torch.device) -> str:
    '''
    Turn a requested backend into the one that will actually be used, warning when
    the request cannot be honoured.

    Parameters
    ----------
    backend : Backend
        Requested backend. 'auto' picks the fastest available for the device.
    device : torch.device
        Device the predictor will run on.

    Returns
    -------
    str
        One of 'torch', 'torch-fp16', 'coreml'.
    '''
    if backend == 'auto':
        if coreml_available():
            return 'coreml'
        return 'torch-fp16' if device.type == 'mps' else 'torch'

    if backend == 'coreml' and not coreml_available():
        logger.warning('CoreML backend requested but unavailable (needs Apple silicon + '
                       'coremltools). Falling back to PyTorch.')
        return 'torch-fp16' if device.type == 'mps' else 'torch'

    if backend == 'torch-fp16' and device.type == 'cpu':
        logger.warning('fp16 is not useful on CPU. Falling back to fp32.')
        return 'torch'

    return backend


class AdsPredictor(nnUNetPredictor):
    '''
    nnU-Net predictor with a selectable execution backend for the forward pass.

    Parameters
    ----------
    backend : Backend, optional
        Execution backend, by default 'auto'.
    **kwargs
        Passed through to `nnUNetPredictor`, e.g. `tile_step_size` or `device`.
    '''

    def __init__(self, backend: Backend = 'auto', **kwargs):
        super().__init__(**kwargs)
        self.requested_backend = backend
        self.backend = None          # resolved in initialize_from_trained_model_folder
        self._coreml_models = {}     # fold index -> MLModel
        self._fold_idx = 0
        self._path_model = None
        self._folds = []

    # ------------------------------------------------------------------ setup

    def initialize_from_trained_model_folder(self,
                                             model_training_output_dir: str,
                                             use_folds,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        '''
        Load the model as nnU-Net does, then prepare the chosen backend.
        '''
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)

        self._path_model = Path(model_training_output_dir)
        self._folds = [str(f) for f in use_folds]
        self.backend = resolve_backend(self.requested_backend, self.device)

        # Deep supervision returns logits at several scales; inference wants only
        # the full-resolution head.
        if hasattr(self.network, 'decoder') and hasattr(self.network.decoder, 'deep_supervision'):
            self.network.decoder.deep_supervision = False

        if self.backend == 'coreml':
            try:
                self._prepare_coreml()
            except Exception as e:
                logger.warning(f'CoreML conversion failed ({type(e).__name__}: {e}). '
                               'Falling back to PyTorch.')
                self.backend = 'torch-fp16' if self.device.type == 'mps' else 'torch'

        if self.backend == 'torch-fp16':
            self.network = self.network.half()

        logger.info(f'Inference backend: {self.backend} | tile step: {self.tile_step_size}')

    def _prepare_coreml(self) -> None:
        '''
        Convert each fold to a cached fp16 CoreML program. Conversion takes a few
        seconds and happens once per checkpoint; afterwards it is a file load.
        '''
        import coremltools as ct

        patch_size = tuple(self.configuration_manager.patch_size)
        shape = (1, len(self.dataset_json['channel_names']), *patch_size)
        fingerprint = _checkpoint_fingerprint(self._path_model, self._folds, patch_size)
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        for idx, params in enumerate(self.list_of_parameters):
            path = cache / f'{self._path_model.name}_{fingerprint}_fold{idx}_fp16.mlpackage'

            if not path.exists():
                logger.info(f'Converting fold {self._folds[idx]} to CoreML (one-time, ~5s)...')
                net = self.network
                net.load_state_dict(params)
                net = net.cpu().eval()
                traced = torch.jit.trace(net, torch.randn(*shape))
                model = ct.convert(
                    traced,
                    inputs=[ct.TensorType(name='x', shape=shape, dtype=np.float32)],
                    outputs=[ct.TensorType(name='logits', dtype=np.float32)],
                    convert_to='mlprogram',
                    compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS14,
                )
                model.save(str(path))
                logger.info(f'Cached converted model at {path}')

            # ComputeUnit.ALL lets CoreML schedule onto the Neural Engine.
            self._coreml_models[idx] = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)

        # The torch network no longer runs the forward pass; keep it off the GPU.
        self.network = self.network.cpu()

    # ------------------------------------------------------------- forward pass

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Run one batch through the active backend.

        Parameters
        ----------
        x : torch.Tensor
            Patch batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, n_classes, H, W), float32, on `self.device`.
        '''
        if self.backend == 'coreml':
            out = self._coreml_models[self._fold_idx].predict(
                {'x': np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)}
            )['logits']
            return torch.from_numpy(np.asarray(out)).to(self.device)

        if self.backend == 'torch-fp16':
            return self.network(x.half()).float()

        return self.network(x)

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        '''
        nnU-Net's mirrored prediction, routed through the selected backend. The set
        of flips averaged over is unchanged.
        '''
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self._forward(x)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self._forward(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Average logits over folds, tracking which fold is active so the CoreML
        backend selects the matching converted model. nnU-Net's version reloads
        `self.network` per fold, which CoreML does not need.
        '''
        if self.backend != 'coreml':
            return super().predict_logits_from_preprocessed_data(data)

        prediction = None
        for idx in range(len(self.list_of_parameters)):
            self._fold_idx = idx
            logits = self.predict_sliding_window_return_logits(data).to('cpu')
            prediction = logits if prediction is None else prediction + logits

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)
        return prediction

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor):
        '''
        Skip nnU-Net's unconditional `network.to(device)` when CoreML handles the
        forward pass, which would otherwise copy 33M unused parameters to the GPU
        for every image.
        '''
        if self.backend == 'coreml':
            original = self.network
            self.network = _NullModule()
            try:
                return super().predict_sliding_window_return_logits(input_image)
            finally:
                self.network = original
        return super().predict_sliding_window_return_logits(input_image)


class _NullModule(torch.nn.Module):
    '''
    Placeholder for the torch network while CoreML performs the forward pass.
    `.to()` and `.eval()` are no-ops; it is never called.
    '''

    def forward(self, *args, **kwargs):
        raise RuntimeError('_NullModule should never be called; the CoreML backend '
                           'handles the forward pass.')
