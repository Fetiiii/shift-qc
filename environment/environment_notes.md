# Environment notes

## Lightweight inspection / conceptual figure path

CPython 3.12.13 and environment/requirements-public.txt use recorded direct
dependency versions from frozen environment/lock.txt; setuptools follows
the frozen pyproject build requirement. NumPy, SciPy,
scikit-learn, pandas, nibabel, SimpleITK and Matplotlib support the selected
code/tests. pytest is the test runner; setuptools builds the package.
Platform-specific CUDA transitive dependencies are not required for this path.
This release does not silently upgrade the scientific pipeline environment.

## Full segmentation path: recorded snapshot, not a fresh machine claim

Frozen source environment/system.json and environment/upstreams.json record:
Python 3.12.13; PyTorch 2.12.1+cu132 (runtime CUDA 13.2);
nnU-Net v2 2.8.1; M3DA 0.2.0 at commit
1728287a76526704193525c0f4714a3d028108a2.
The recorded GPU was NVIDIA GeForce RTX 5070 Laptop GPU, driver 610.57.04,
local CUDA toolkit 13.3. OS was CachyOS Linux x86_64, kernel 7.1.8-1-cachyos,
glibc 2.44. These are historical environment records within the frozen export
source, not measurements taken during TASK-045.

The full pipeline requires compatible GPU/CUDA tooling, standard nnU-Net
planning, obtained source inputs and cleared frozen provenance/splits.
Weights and medical inputs are withheld. Retraining is expensive, and
bitwise GPU reproducibility is not established. The original source lock
remains authoritative; this portable subset does not freeze unrelated
platform-specific transitive packages.
