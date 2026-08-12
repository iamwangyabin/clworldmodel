# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="clworldmodel ARROW-50"
LABEL org.opencontainers.image.description="ARROW-50 reference environment"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/arrow \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch, torchvision, and torchaudio come from BASE_IMAGE. Keeping them out
# of this pip transaction prevents an unnecessary request to download.pytorch.org.
RUN python -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        ale-py==0.11.1 \
        gymnasium==1.1.1 \
        gymnasium-notices==0.0.1 \
        matplotlib==3.10.0 \
        numpy==1.26.4 \
        opencv-python==4.11.0.86 \
        sortedcontainers==2.4.0 \
        tensorboard==2.13.0 \
        tensorboard-data-server==0.7.2 \
        tianshou==0.5.1 \
        tqdm==4.67.1

RUN groupadd --gid "${GROUP_ID}" arrow \
    && useradd --create-home --uid "${USER_ID}" --gid "${GROUP_ID}" --shell /bin/bash arrow

WORKDIR /workspace/clworldmodel
COPY --chown=arrow:arrow . .

RUN python scripts/verify_arrow_environment.py

USER arrow

CMD ["bash"]
