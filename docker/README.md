# PIE imaging container

`docker/Dockerfile` packages PIE's imaging stack as it produced the validated numbers, for review and for running
inside secure enclaves. `docker/pie-imaging.def` converts the Docker image for Apptainer.

| Component | Version | Source |
|---|---|---|
| Python environment | Python 3.12.3, torch 2.7.1+cu128 (or +cpu), antspyx 0.6.3, dipy 1.12.1, SimpleITK 2.5.5, nibabel 5.4.2, nilearn 0.14.1, numpy 2.3.5, pandas 3.0.3, scipy 1.15.3, neuroCombat 0.2.12, truenet 2.0.2 | `requirements-imaging.lock` / `requirements-imaging-cpu.lock` (hashed) |
| FastSurfer | 2.6.0-dev0, Deep-MI/FastSurfer@d0d7d39 (identical to the checkout the T1 validation ran on), VINN + CC checkpoints baked in | GitHub archive |
| FreeSurfer | 7.4.1 (ubuntu22 build 20230614-7eb8460) | surfer.nmr.mgh.harvard.edu |
| FSL | 6.0.7.23 | `fslinstaller.py -V 6.0.7.23` |
| ANTs (command line) | 2.6.5 (sha256-checked); PIE itself uses antspyx | GitHub release |
| MRtrix3 | 3.0.4 | Ubuntu 24.04 package |
| dcm2niix | v1.0.20260724 | PyPI wheel, in the lock |

## Build

From the repository root (the root `.dockerignore` limits the build context to `pie/`, `LICENSE` and the locks, so
nothing from `PPMI/`, `Imaging/` or other local data can enter the image):

```bash
docker build -f docker/Dockerfile -t pie-imaging .                                   # NVIDIA GPU (CUDA 12.8)
docker build -f docker/Dockerfile --build-arg TORCH_BACKEND=cpu -t pie-imaging:cpu .  # no GPU
```

The build downloads FreeSurfer (9.5 GB), FSL, ANTs, the Python wheels and the FastSurfer checkpoints. Expect an image
of roughly 30-40 GB (not yet measured) and leave about twice that free while building.

## Run

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v /path/to/Imaging:/data:ro \
    -v /path/to/derived:/out \
    -v /path/to/pie-home:/home/pie \
    -v /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    pie-imaging \
    python -m pie.imaging.run --zips /data/MRI_First_Study.zip --ppmi-dir /data/PPMI --work-dir /out
```

- `--gpus all` needs the NVIDIA Container Toolkit and a driver new enough for CUDA 12.8 (R570 or later). Without a
  GPU, leave it out and pass `--device cpu` to PIE (use the `cpu` build if the host has no NVIDIA driver at all).
- The FreeSurfer licence (free, per user: https://surfer.nmr.mgh.harvard.edu/registration.html) is only read by
  `run --etiv`, `run --long` and `fastsurfer.surfaces`. It is never part of the image.
- `/home/pie` holds PIE's download cache, `~/.cache/pie` (TemplateFlow templates, the Biondetti and JHU atlases,
  snceg weights). Mount it so downloads and the cached atlas registrations persist between runs.
- Any `pie.imaging` module runs the same way (`python -m pie.imaging.dwi ...`, `python -m pie.imaging.nm ...`).

## Apptainer

```bash
apptainer build pie-imaging.sif docker/pie-imaging.def            # from the local Docker image
# or, where there is no Docker: docker save pie-imaging -o pie-imaging.tar on the build machine, then
apptainer build pie-imaging.sif docker-archive://pie-imaging.tar

apptainer exec --nv --cleanenv \
    -B /path/to/Imaging:/data:ro -B /path/to/derived:/out \
    -B /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt \
    pie-imaging.sif python -m pie.imaging.run --zips /data/MRI_First_Study.zip --ppmi-dir /data/PPMI --work-dir /out
```

`--nv` exposes the GPU. `--cleanenv` stops host variables (`PIE_*`, `FSLDIR`, `PYTHONPATH`) overriding the image's.
Apptainer binds your host home, so the cache is your own `~/.cache/pie`.

## Secure enclaves (PPP's PEP)

The Personalized Parkinson Project's PEP environment does not let data leave, so PIE has to run inside it, as this
container. Build the image (or the `.sif`) outside, bring the image file in through PEP's import procedure and run it
there. The enclave has no internet, so stage beforehand what PIE otherwise fetches on first use:

- `~/.cache/pie`: run the steps you need once on a networked machine with the same cache mount, then import the folder
  and mount it as above;
- pretrained weights for `embed` and `cnn --pretrained`: `third_party/weights/` (see its `WEIGHTS.md`), mounted
  anywhere and named with `-e PIE_WEIGHTS_DIR=/weights`.

Only outputs that PEP's export rules allow should leave the enclave.

Not in the image: the snceg environment for `nm_native --snceg` (torch 2.0.1 + fastMONAI; `PIE_SNCEG_PYTHON`),
fMRIPrep (run its own pinned image; `pie.imaging.fmriprep`), the brain viewer's built frontend, TrUE-Net weights,
`pie-clean`, and `lifelines` (it requires pandas < 3; `pie.stats.survive` belongs to the general PIE environment).

## Licences

PIE is MIT; FastSurfer and ANTs Apache-2.0; MRtrix3 MPL-2.0; dcm2niix BSD. **FSL is licensed for non-commercial use
only** (https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html): commercial users need a licence from Oxford University
Innovation, and anyone distributing an image that contains FSL should check its terms first. FreeSurfer's licence key
is per user and must be supplied at run time.

## Updating a pin

Python packages: edit `requirements-imaging.in`, then rerun the command in the header of each lock. Tools: the `ARG`s at
the top of the Dockerfile (FreeSurfer and MRtrix3 are pinned in their own `RUN` lines). FastSurfer, FreeSurfer, FSL and torch versions all move PIE's numbers, so re-run
the validation before changing them.
