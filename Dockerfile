# Camaron Audio — single source of truth, parameterized by RUNTIME (cpu|cuda|rocm).
#
#   docker build -t camaron-audio .                                   # CPU (default)
#   docker build --build-arg RUNTIME=cuda  -t camaron-audio .         # NVIDIA
#   docker build --build-arg RUNTIME=rocm  -t camaron-audio .         # AMD
#
# The three `FROM ... AS cpu/cuda/rocm` stages are candidate bases; `FROM ${RUNTIME}`
# selects exactly one, so building the CPU image never pulls the CUDA/ROCm bases.

ARG RUNTIME=cpu

# --- candidate base images ---------------------------------------------------
FROM python:3.12-slim AS cpu

# The cuda/rocm bases don't ship a Python new enough for this code (3.11+); add 3.12.
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS cuda
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
 && python3.12 -m ensurepip --upgrade

FROM rocm/rocm-dev-ubuntu-22.04 AS rocm
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip \
 && python3.12 -m ensurepip --upgrade

# --- build on the selected base ---------------------------------------------
FROM ${RUNTIME} AS image
ARG RUNTIME
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
WORKDIR /opt/camaron

# Leaner to copy metadata + source only.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the service + TTS phonemization deps, then swap in the onnxruntime build
# matching the runtime (cpu is the dep default; gpu/rocm replace it).
RUN PIP="python3.12 -m pip" && \
    $PIP install --upgrade pip && \
    $PIP install ".[tts]" && \
    if [ "$RUNTIME" = "cuda" ]; then \
      $PIP uninstall -y onnxruntime && $PIP install onnxruntime-gpu; \
    elif [ "$RUNTIME" = "rocm" ]; then \
      $PIP uninstall -y onnxruntime && $PIP install onnxruntime-rocm; \
    fi

# Models are mounted, never baked in. Voice style tables are .npy (torch-free).
ENV CAMARON_MODEL_PATH=/models
EXPOSE 8080
VOLUME ["/models"]
CMD ["python3.12", "-m", "src", "--port", "8080"]
