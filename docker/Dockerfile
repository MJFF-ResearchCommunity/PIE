# syntax=docker/dockerfile:1
# PIE imaging stack, pinned to the environment that produced PIE's validated numbers. Build from the repo root:
#   docker build -f docker/Dockerfile -t pie-imaging .
# Everything sits where PIE's defaults look (<repo>/venv_imaging, <repo>/third_party/FastSurfer, $FSLDIR), so the only
# run-time input besides data is the FreeSurfer licence mount. See docker/README.md.
FROM ubuntu:noble-20260917@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# cu128 = the validated torch 2.7.1+cu128 (requirements-imaging.lock); cpu = requirements-imaging-cpu.lock, for hosts
# without an NVIDIA GPU (FastSurfer on CPU is slower and its segmentations differ slightly from the GPU ones)
ARG TORCH_BACKEND=cu128
# Deep-MI/FastSurfer dev at 2.6.0-dev0: the snapshot PIE's T1 validation ran on (the dev machine's third_party/FastSurfer
# matches this commit file for file). The v2.6.0 tag that scripts/setup_imaging.sh now clones moved to Python 3.14 and
# torch 2.14 and has not been validated; switch with --build-arg FASTSURFER_REF=v2.6.0 only after re-validation.
ARG FASTSURFER_REF=d0d7d39369552f8f8b2f14c06d85a3516d06e4cb
ARG FSL_VERSION=6.0.7.23
ARG ANTS_VERSION=2.6.5
ARG ANTS_SHA256=9f6a2c8881d6c94ea17171c0502e55cb898ad9afd50e4e23fde9a8c0997aa04b

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

# python3 is 3.12.3, as in venv_imaging; tcsh, bc, libgomp1, perl and the X libs are FreeSurfer's (talairach_avi needs
# tcsh); libcairo2 is cairosvg's (fmri_qc); git fetches the truenet pin; MRtrix3 3.0.4 is the dev machine's package.
# Only MRtrix3 is version-pinned: the base image is pinned by digest and the rest are system libraries left on noble's
# security updates.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc bzip2 ca-certificates curl git libcairo2 libglu1-mesa libgomp1 libxmu6 libxt6 mrtrix3=3.0.4-1build2 \
        perl python3 tcsh unzip \
    && rm -rf /var/lib/apt/lists/*

# FreeSurfer 7.4.1, the ubuntu22 build the dev machine runs (no checksum is published, so the build stamp is checked).
# Its licence is per user: mounted at run time at $PIE_FS_LICENSE, never baked in.
RUN curl -fsSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
        | tar -xzf - -C /opt \
    && grep -qx 'freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460' /opt/freesurfer/build-stamp.txt

# FSL (non-commercial licence, see README). --cuda fixes the CUDA build of eddy_cuda whatever GPU the build host has.
RUN curl -fsSL -o /tmp/fslinstaller.py https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py \
    && python3 /tmp/fslinstaller.py -d /opt/fsl -V "$FSL_VERSION" --skip_registration --cuda 12.8 --no_self_update \
    && grep -q "^$FSL_VERSION" /opt/fsl/etc/fslversion \
    && rm -rf /tmp/fslinstaller.py /opt/fsl/pkgs

# ANTs command-line tools (PIE itself calls ANTs through antspyx 0.6.3 from the lock)
RUN curl -fsSL -o /tmp/ants.zip \
        "https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-ubuntu-24.04-X64-gcc.zip" \
    && echo "${ANTS_SHA256}  /tmp/ants.zip" | sha256sum -c - \
    && unzip -q /tmp/ants.zip -d /tmp/ants && mv /tmp/ants/*/ /opt/ants && rm -rf /tmp/ants /tmp/ants.zip

# Python environment from the lock (dcm2niix v1.0.20260724 comes with it, at PIE's default <repo>/venv_imaging/bin)
COPY --from=ghcr.io/astral-sh/uv:0.7.19@sha256:2dcbc74e60ed6d842122ed538f5267c80e7cde4ff1b6e66a199b89972496f033 /uv /usr/local/bin/uv
COPY requirements-imaging.lock requirements-imaging-cpu.lock /opt/pie/
RUN lock=/opt/pie/requirements-imaging.lock; \
    if [ "$TORCH_BACKEND" = cpu ]; then lock=/opt/pie/requirements-imaging-cpu.lock; fi; \
    UV_PYTHON_DOWNLOADS=never uv venv /opt/pie/venv_imaging --python /usr/bin/python3 \
    && UV_NO_CACHE=1 uv pip sync --python /opt/pie/venv_imaging/bin/python --torch-backend "$TORCH_BACKEND" "$lock"

# FastSurfer and the checkpoints PIE's segmentation stream uses (VINN + corpus callosum), so nothing downloads at run time
WORKDIR /opt/pie/third_party/FastSurfer
RUN curl -fsSL "https://github.com/Deep-MI/FastSurfer/archive/${FASTSURFER_REF}.tar.gz" | tar -xzf - --strip-components=1 \
    && PYTHONPATH=. /opt/pie/venv_imaging/bin/python FastSurferCNN/download_checkpoints.py --vinn --cc

COPY LICENSE /opt/pie/
COPY pie /opt/pie/pie

ENV PYTHONPATH=/opt/pie \
    PATH=/opt/pie/venv_imaging/bin:/opt/fsl/share/fsl/bin:/opt/ants/bin:$PATH \
    FSLDIR=/opt/fsl FSLOUTPUTTYPE=NIFTI_GZ \
    PIE_FREESURFER_HOME=/opt/freesurfer PIE_FS_LICENSE=/opt/freesurfer/license.txt \
    HOME=/home/pie
# HOME holds PIE's download cache (~/.cache/pie: atlases, snceg weights); mount it to keep or pre-seed it
RUN mkdir -m 1777 /home/pie
LABEL org.opencontainers.image.title="PIE imaging" \
      org.opencontainers.image.source="https://github.com/MJFF-ResearchCommunity/PIE" \
      org.opencontainers.image.licenses="MIT (PIE); FSL and FreeSurfer under their own licences" \
      pie.fastsurfer="${FASTSURFER_REF}" pie.freesurfer="7.4.1" pie.fsl="${FSL_VERSION}" pie.ants="${ANTS_VERSION}" \
      pie.mrtrix3="3.0.4" pie.dcm2niix="1.0.20260724" pie.torch-backend="${TORCH_BACKEND}"
WORKDIR /work
CMD ["python", "-m", "pie.imaging.run", "--help"]
