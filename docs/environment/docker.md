# ARROW-50 Docker environment

## Image contract

The default base image is
`pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime`. It provides Python 3.10,
PyTorch 2.3.0, torchvision 0.18.0, torchaudio 2.3.0, CUDA 11.8 runtime, and
cuDNN 8. The smaller runtime image is sufficient because ARROW does not build
custom CUDA extensions.

The build installs non-PyTorch dependencies through the TUNA PyPI mirror.
`ale-py==0.11.1` supplies the ROMs, so the image never invokes AutoROM or
contacts GitHub Gist. The build fails unless all six ARROW Atari environments
can be reset headlessly.

## Build

From the repository root on a Linux x86_64 machine:

If the GPU provider first requires a platform-image choice, select its
`CUDA11.8-PyTorch2.0.1` image with Python 3.10 and Ubuntu 22.04. That image is
only the Docker host and fallback environment; its PyTorch 2.0.1 installation
does not enter the container. The Docker image below still supplies the frozen
PyTorch 2.3.0/cu118 runtime.

```bash
docker build \
  --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag clworldmodel:arrow-2.3.0-cu118 \
  .
```

If Docker Hub is inaccessible, first mirror the exact official base image to a
registry you trust, then pass its address without changing its tag contents:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=REGISTRY/PATH/pytorch:2.3.0-cuda11.8-cudnn8-runtime \
  --tag clworldmodel:arrow-2.3.0-cu118 \
  .
```

Use `--build-arg PIP_INDEX_URL=...` to choose another HTTPS PyPI mirror. Match
`USER_ID` and `GROUP_ID` to the server account when mounted outputs must retain
host ownership.

## Verify on GPU

The image build verifies package versions and ROMs without requiring a GPU.
After the build, verify NVIDIA passthrough:

```bash
docker run --rm \
  --gpus all \
  clworldmodel:arrow-2.3.0-cu118 \
  python scripts/verify_arrow_environment.py --require-cuda
```

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. CUDA
does not need to be installed separately on the host beyond the driver/toolkit
requirements.

## Run ARROW-50

Persist experiment outputs on the host:

```bash
mkdir -p runs
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --volume "$(pwd)/runs:/workspace/clworldmodel/runs" \
  clworldmodel:arrow-2.3.0-cu118 \
  python scripts/run_arrow_ar50_atari.py --seed 0
```

Do not run AutoROM in the container and do not mount or commit a separate ROM
directory.

## Restricted platform builder

Some GPU providers select the base image outside the Dockerfile and prohibit
`FROM`, `EXPOSE`, `CMD`, and `ENTRYPOINT`. For that builder, select the
CUDA 11.8 / PyTorch 2.0.1 / Python 3.10 / Ubuntu 22.04 base and paste
`Dockerfile.platform` into its editor. It upgrades the selected base to the
frozen PyTorch 2.3.0/cu118 wheels through the Alibaba Cloud mirror, installs
the remaining dependencies through TUNA, and verifies all six ROMs during the
build. No uploaded auxiliary file is required.
