# docker/base.Dockerfile | dramatizer-base: CUDA, Python 3.10, the packages in requirements.txt and CosyVoice's
# code. About 8-10 GB, so it is built rarely (when requirements.txt or the CosyVoice commit changes), by hand on
# the GPU computer, and pushed to ghcr.io; the app image (docker/app.Dockerfile) is built on it by its digest.
#
#   docker build -f docker/base.Dockerfile -t ghcr.io/ahmed-sadeq-hussein/dramatizer-base:2026-11-23 .
#
# No model weights, voices, tokens or data are in it (Milestone 4, problems 1-3): they are mounted at run time.
# Two stages: the compilers needed to build a few packages (pyworld, openai-whisper) stay in the first.

# CUDA 12.1 + cuDNN 8 to match torch 2.3.1+cu121. Pinned by digest: the tag can be moved, the digest cannot.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30

FROM ${CUDA_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The same order as the set-up steps at the end of requirements.txt
RUN pip install "setuptools<81" wheel \
    && pip install --no-build-isolation openai-whisper==20231117 \
    && pip install --no-deps resemblyzer==0.1.4 deepfilternet==0.5.6 deepfilterlib==0.5.6
# deepspeed and TensorRT are left out: CosyVoice only uses them when asked (load_trt / fp16 engines), we never
# do, and together they add several GB and need the CUDA compiler.
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE "^(deepspeed|tensorrt)" /tmp/requirements.txt > /tmp/requirements-image.txt \
    && pip install -r /tmp/requirements-image.txt

# CosyVoice is not on PyPI: its code at one fixed commit, with the Matcha-TTS submodule that commit points to
ARG COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
RUN git clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && git -C /opt/CosyVoice checkout --quiet ${COSYVOICE_COMMIT} \
    && git -C /opt/CosyVoice submodule update --init --recursive --depth 1 \
    && rm -rf /opt/CosyVoice/.git /opt/CosyVoice/third_party/*/.git /opt/CosyVoice/asset /opt/CosyVoice/docker


FROM ${CUDA_IMAGE}
ARG COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH VOICE_MODEL_DIR=/opt/CosyVoice
# sox and ffmpeg: CosyVoice and Whisper call them; libsndfile: soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 sox libsox-fmt-all ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/CosyVoice /opt/CosyVoice
LABEL org.opencontainers.image.source="https://github.com/Ahmed-Sadeq-Hussein/Dramatizer" \
      org.opencontainers.image.description="Dramatizer 2 base: CUDA 12.1, Python 3.10, requirements.txt, CosyVoice" \
      dramatizer.cosyvoice.commit="${COSYVOICE_COMMIT}"
