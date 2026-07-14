# AxonDeepSeg segmentation service.
#
# Build for the platform you will RUN on, not the one you build on. GPU instances
# (g4dn/g5) are x86_64, so on an ARM Mac you must either build on an x86 host (e.g.
# the EC2 instance itself) or cross-build:
#
#   docker build --platform linux/amd64 -t axondeepseg-api .
#
# Run (see --shm-size note at the bottom -- it is not optional):
#
#   docker run --gpus all --shm-size=1g -p 8000:8000 axondeepseg-api
#
# No GPU? It falls back to CPU automatically; drop --gpus all.

FROM python:3.11-slim

# libgl/libglib: OpenCV-style native deps pulled in by scikit-image/imageio.
# curl: used by the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch on its own layer, before the source is copied. It is by far the
# largest thing in the image (the linux wheel bundles the CUDA runtime), and pinning
# it here means editing application code does not re-download several GB.
# The version must satisfy the "torch<2.4.0" pin in pyproject.toml.
#
# For a CPU-only image (much smaller, no GPU), instead use:
#   RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.3.1

# Headless install: no [gui] extra, so no napari and no Qt. The API imports none of
# it, and PyQt5 has no linux-aarch64 wheel, so this is also what makes ARM possible.
COPY pyproject.toml README.md ./
COPY AxonDeepSeg ./AxonDeepSeg
RUN pip install --no-cache-dir .

# Bake the model weights into the image. Doing this at build time rather than at boot
# is deliberate: a scale-to-zero container must not spend its cold start downloading
# 256 MB, and download_model() is unsafe at runtime anyway -- it sys.exit()s on
# failure and unzips into the current working directory.
#
RUN download_model -m generalist
RUN download_model -m dedicated-BF
RUN download_model -m dedicated-SEM
RUN download_model -m unmyelinated-TEM

# Drop privileges. Done after the installs so the model lands in root-owned
# site-packages and the service only ever reads it.
RUN useradd --create-home --uid 1000 ads
USER ads

EXPOSE 8000

# /ready returns 503 until the model weights are present, so an orchestrator will not
# route traffic to a replica that cannot actually serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

# Single process, deliberately. The job store is in-memory and inference is guarded by
# an in-process lock, so multiple uvicorn workers would hand a client a job id that the
# next worker does not know about. Do NOT add --workers.
#
# IMPORTANT: run with --shm-size=1g (or --ipc=host). nnU-Net spawns ~8 worker processes
# and Docker's default /dev/shm is 64 MB, which surfaces as bus errors / killed workers
# rather than an obvious out-of-memory message.
CMD ["ads_server"]
