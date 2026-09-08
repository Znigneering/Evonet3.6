# EvoNet for Python 3.6

PyTorch experiments for few-shot image classification, featuring ResNet-12 and ResNet-50 backbones with evolving connections, Prototypical Networks, and genetic programming-based additive boosting with a TIM (Transductive Information Maximization) objective.

This repository targets **64-bit Python 3.6**. Dependencies are pinned in [requirements.txt](requirements.txt); PyTorch requires **Python 3.6.2 or a later 3.6 patch release**. Run all commands below from the repository root.

## Features

- **Evolving network connections:** convolutional and linear layers use masks to control active connections, prune weights by magnitude, and select connections to reactivate using gradient signals. `Evonet.py` and `Evonet50.py` use a straight-through estimator to propagate gradients to inactive connections.
- **Few-shot classification:** each episode computes class prototypes from the support set and classifies query features using squared Euclidean distances to those prototypes.
- **Two backbones:** `Evonet.py` implements ResNet-12; `Evonet50.py` implements ResNet-50 with `EpisodeBatchNorm2d`, which uses statistics from the current batch.
- **Two dataset workflows:** automatically downloaded Omniglot data and a local `OmnImage84_100` dataset organized into class folders.
- **TIM boosting:** `additive_boosting_tim.py` searches for additive feature transformations using genetic programming and accelerates TIM loss computation with Numba. The objective uses support labels and query features; query labels are not used to compute the optimization objective.

`density` controls the initial fraction of active connections and is not guaranteed to remain constant after evolution. Masks are applied to ordinary dense tensors, so a lower active-connection ratio does not imply a proportional reduction in runtime or memory use.

## Repository structure

| File | Purpose |
| --- | --- |
| `Evonet.py` | Evolving ResNet-12, feature projection layer, and Omniglot episode sampler |
| `Evonet50.py` | Evolving ResNet-50, Episode BatchNorm, and Omniglot episode sampler |
| `train_debug.py` | ResNet-12 training on Omniglot; requires the return-value unpacking adjustment described below |
| `train_omniglot_evonet.py` | **ResNet-50** training on Omniglot with a projection dimension of 640 |
| `train_omniimage_evonet50.py` | ResNet-50 training on local OmniImage data with a projection dimension of 64 |
| `omni_image_sampler.py` | Class splits, image preprocessing, and episode sampling for local data |
| `evaluation_evonet.py` | Evaluation of ResNet-12 ProtoNet weights on Omniglot |
| `evaluation_evonet50.py` | Evaluation of ResNet-50 ProtoNet weights on OmniImage |
| `additive_boosting_tim.py` | TIM loss, genetic programming-based boosting, and a synthetic-data smoke test |
| `transductive_tim_omniglot.py` | ProtoNet / TIM boosting comparison on Omniglot; requires checkpoint-loading adaptation |

The scripts do not expose a shared command-line argument parser. Configure learning rates, episode counts, data directories, model dimensions, and output paths directly in the corresponding scripts. Flags such as `--epochs` are not supported.

## Installation

### 1. Get the code and create a Python 3.6 environment

```bash
git clone https://github.com/Znigneering/Evonet3.6.git
cd Evonet3.6
conda create -n evonet36 python=3.6 -y
conda activate evonet36
python --version
```

If Python 3.6 is already installed, you can use `venv` instead. On Linux / macOS:

```bash
python3.6 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.6 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Pin the installation tools

Run this command inside the newly created and activated environment:

```bash
python -m pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"
```

These versions support Python 3.6 and avoid upgrading the installation tools to releases that require a newer interpreter. The commands below use `python -m pip` to install packages into the active interpreter's environment.

### 3. Install project dependencies

To use the default builds available on PyPI:

```bash
python -m pip install -r requirements.txt
```

To select a specific CPU or CUDA build, choose one of the following alternatives in a fresh environment. Install the matching `torch` / `torchvision` builds first, then install the complete requirements.

**CPU only, Linux / Windows:**

```bash
python -m pip install "torch==1.10.2+cpu" "torchvision==0.11.3+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install -r requirements.txt
```

**CUDA 11.3, Linux / Windows:**

```bash
python -m pip install "torch==1.10.2+cu113" "torchvision==0.11.3+cu113" -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install -r requirements.txt
```

CUDA builds require a compatible NVIDIA GPU and driver. Both `+cpu` and `+cu113` builds satisfy the version pins without suffixes in requirements. Intel macOS can use the default installation method; the `+cpu` / `+cu113` commands above target Linux / Windows. This dependency configuration does not cover native Apple Silicon environments. Available builds are listed in the official PyTorch [CPU wheel index](https://download.pytorch.org/whl/cpu/torch_stable.html) and [CUDA 11.3 wheel index](https://download.pytorch.org/whl/cu113/torch_stable.html).

### 4. Verify the installation

```bash
python -m pip check
python -c "import sys, torch, torchvision, numpy, PIL, tqdm, numba, llvmlite; print('Python:', sys.version); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('Pillow:', PIL.__version__); print('tqdm:', tqdm.__version__); print('Numba:', numba.__version__); print('llvmlite:', llvmlite.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
```

Run the TIM / boosting smoke test without downloading a dataset or providing a checkpoint:

```bash
python additive_boosting_tim.py
```

This script generates random data, computes the TIM loss, and runs a small boosting experiment. It prints `Fit complete.` on successful completion. The first invocation includes Numba compilation overhead and should not be used as a steady-state runtime measurement.

Verification has been performed in an isolated Windows x64 environment with CPython 3.6.8, PyTorch `1.10.2+cpu`, and torchvision `0.11.3+cpu`. Checks covered dependency installation, `pip check`, compilation of all Python source files, forward passes and connection evolution for both backbones using random inputs, episode sampling from synthetic images, and the TIM / boosting smoke test above. This verification does not cover CUDA execution, full training on real datasets, or reproduction of classification accuracy.

### Dependency versions

| Dependency | Pinned version | Purpose |
| --- | --- | --- |
| PyTorch | `1.10.2` | Networks, automatic differentiation, and optimizers |
| torchvision | `0.11.3` | Omniglot dataset and image transforms; paired with the pinned PyTorch version |
| NumPy | `1.19.5` | Episode sampling and numerical computations for boosting |
| Pillow | `8.4.0` | Image loading; imported as `PIL` in the code |
| tqdm | `4.64.1` | Training and evaluation progress bars |
| Numba / llvmlite | `0.53.1` / `0.36.0` | JIT compilation of the TIM loss |

The requirements also pin runtime transitive dependencies: `setuptools`, `typing-extensions`, the Python 3.6 dependencies `dataclasses` / `importlib-resources` / `zipp`, and `colorama` on Windows. Release metadata is available for [PyTorch 1.10.2](https://pypi.org/project/torch/1.10.2/), [torchvision 0.11.3](https://pypi.org/project/torchvision/0.11.3/), [Numba 0.53.1](https://pypi.org/project/numba/0.53.1/), and [tqdm 4.64.1](https://pypi.org/project/tqdm/4.64.1/).

## Data preparation

### Omniglot

`OmniglotBoosterTaskSampler` uses `torchvision.datasets.Omniglot` with `download=True`, so the first run requires an internet connection. The dataset root is `../data`, relative to the current working directory.

- Training uses `background=True`; validation and evaluation use `background=False`.
- Images are resized to 84 x 84, converted to tensors, inverted, and repeated from one channel to three channels.
- Each episode returns support images and labels, followed by query images and labels. Labels are remapped within the episode to `0` through `n_way - 1`.
- Each class requires `k_shot + n_query` images. If too few images are available, the Omniglot sampler samples with replacement, which can introduce duplicate images within or across the support and query sets.

### OmniImage / `OmnImage84_100`

Prepare this dataset separately; the repository includes neither the data nor a downloader. By default, the training and evaluation scripts look for `data/OmnImage84_100` next to the repository directory. Use the directory spelling shown in the code:

```text
workspace/
├── Evonet3.6/
│   ├── train_omniimage_evonet50.py
│   ├── evaluation_evonet50.py
│   └── ...
└── data/
    └── OmnImage84_100/
        ├── class_001/
        │   ├── image_001.jpg
        │   └── image_002.jpg
        ├── class_002/
        │   └── ...
        └── ...
```

Each immediate subdirectory of the dataset root represents a class. Place images directly inside the class folders; supported extensions are `.png`, `.jpg`, `.jpeg`, and `.bmp`. Do not add an extra `train/`, `val/`, or `test/` directory level.

The sampler sorts the class names, shuffles them with the fixed seed `42`, and assigns 80% of classes to training and 20% to testing by default. `val` and `test` share the same held-out classes; there is no separate validation split. Classes with fewer than `k_shot + n_query` images are filtered out, and the selected split must retain at least `n_way` valid classes. Images are converted to RGB, resized to 84 x 84, and converted to tensors in `[0, 1]`.

Check data loading before training:

```bash
python omni_image_sampler.py
```

This smoke test samples a 5-way, 1-shot episode with 5 query images per class, so each class needs at least 6 images. Set `DATA_ROOT` in the training and evaluation scripts to change the data location; the sampler's smoke test uses its default relative path.

## Training and evaluation

`n_way` is the number of classes per episode, `k_shot` is the number of support images per class, and `n_query` is the number of query images per class. The table below shows the defaults in the source code. If GPU memory is insufficient, first reduce the number of classes or query images per class.

| Training script | Dataset / backbone | Training episodes | Training way / shot / query | Validation way / shot / query | `z_dim` / initial `density` |
| --- | --- | --- | --- | --- | --- |
| `train_debug.py` | Omniglot / ResNet-12 | 5,000 | 60 / 1 / 15 | 5 / 1 / 15 | 64 / 0.5 |
| `train_omniglot_evonet.py` | Omniglot / ResNet-50 | 10,000 | 80 / 1 / 1 | 20 / 1 / 1 | 640 / 1.0 |
| `train_omniimage_evonet50.py` | OmniImage / ResNet-50 | 100,000 | 20 / 1 / 1 | 5 / 1 / 1 | 64 / 1.0 |

All three scripts use a learning rate of `1e-3`. In table order, their pruning thresholds and evolution intervals are `0.001` / 500, `0.01` / 50, and `0.001` / 1,000 episodes. For an initial end-to-end check, reduce `N_TRAIN_EPISODES`, validation counts, and episode sizes in the scripts.

### OmniImage: ResNet-50 (still in experiments)

After preparing the data, start training:

```bash
python train_omniimage_evonet50.py
```

The checkpoint is saved to:

```text
Baseline/evo_omniimage/evonet50_omniimage_100k-epoch_1e-3-lr_1000-evo-interval.pth
```

Before evaluation, check `DEVICE` in `evaluation_evonet50.py`. The current script selects `cuda:1` when CUDA is available, which refers to the second visible GPU. On a system with only one visible GPU, change it to:

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

Then run:

```bash
python evaluation_evonet50.py
```

Evaluation defaults to 600 episodes of 20-way, 1-shot classification with 1 query image per class. The test split must contain at least 20 valid classes.

### Omniglot: ResNet-12

**First update the evolution return-value unpacking in `train_debug.py`.** `evolve_all()` in `Evonet.py` returns four values, while the training script currently unpacks only three. The first evolution step, at episode 500 by default, will raise `ValueError: too many values to unpack`. Replace the corresponding statement with:

```python
pruned, grown, _, _ = backbone.evolve_all(prune_threshold=PRUNE_THRESHOLD)
```

After making this adjustment, run:

```bash
python train_debug.py
python evaluation_evonet.py
```

The training script saves `Baseline/evo_omniglot/evonet_test.pth` whenever validation accuracy improves on the previous best. The evaluation script reads the same path and defaults to 600 episodes of 20-way, 1-shot classification with 1 query image per class.

### Omniglot: ResNet-50

```bash
python train_omniglot_evonet.py
```

Although its filename does not include `50`, this script imports its network from `Evonet50.py`, uses `z_dim=640`, and saves `Baseline/evo_omniglot/evonet50_test.pth` at the end of training.

The repository does not currently provide an evaluation entry point that directly matches this training configuration: `evaluation_evonet.py` uses ResNet-12, while `evaluation_evonet50.py` uses OmniImage data and `z_dim=64`. To evaluate this checkpoint, configure **ResNet-50, the Omniglot sampler, `z_dim=640`, and the matching checkpoint path** together. Changing only the checkpoint filename is insufficient.

### Checkpoints and evaluation protocol

Training scripts save the `state_dict` of the `PrototypicalNetworks` wrapper, with parameter names prefixed by `backbone.`. Loading requires the matching backbone type and projection dimension; `strict=False` does not resolve shape mismatches for parameters with the same name.

Both `evaluation_evonet*.py` scripts concatenate support and query images before feature extraction and use normalization statistics from the current batch. This is transductive evaluation because it uses information from the query distribution. Validation during training processes support and query images separately. Record these settings when comparing accuracy results.

Checkpoints under `Baseline/` are generated by the scripts and excluded by `.gitignore`; no pretrained weights are included in the repository. Each script writes to a fixed `SAVE_PATH`, so use separate paths when keeping results from multiple runs.

## TIM boosting comparison

`transductive_tim_omniglot.py` compares prototype classification with TIM boosting on Omniglot. The defaults are 10 episodes of 5-way, 5-shot classification with 15 query images per class. A separate boosting model is fitted for each episode.

**Adapt checkpoint loading before running this experiment.** The script treats the return value of `torch.load()` as a complete network and calls `.eval()` and `.encoder()` on it. The training scripts in this repository save a `state_dict`, so directly using a checkpoint produced by `train_debug.py` raises an error because an `OrderedDict` has no `.eval()` method.

For the ResNet-12 `evonet_test.pth` described above, replace the loading section in `main()`, from `print("Loading Baseline Model...")` through `model.eval()`, with the following code to restore the feature network:

```python
from Evonet import EvolvingProtoNet as ResNet12ProtoNet

state_dict = torch.load(checkpoint_path, map_location='cpu')
backbone_state = {
    key[len('backbone.'):]: value
    for key, value in state_dict.items()
    if key.startswith('backbone.')
}
model = ResNet12ProtoNet(x_dim=3, z_dim=64, density=0.5)
model.load_state_dict(backbone_state, strict=True)
model = model.to(DEVICE)
model.eval()
```

After adapting the loader and preparing the matching checkpoint, run:

```bash
python transductive_tim_omniglot.py
```

The experiment reports ProtoNet accuracy, boosting accuracy, and their difference. Results depend on the trained checkpoint, sampled tasks, and experiment configuration. The repository does not include a benchmark table with reproducible accuracy results.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No matching distribution found` | Confirm that the interpreter is 64-bit Python 3.6, at patch version 3.6.2 or later, and pip is 21.3.1; use the installation commands for your platform |
| `No module named PIL` | The package name is `Pillow`, which is included in requirements |
| `Dataset root not found` / too few valid classes | Check `DATA_ROOT`, class-folder layout, images per class, and `n_way` after splitting |
| `Checkpoint not found` | Train the matching model first and check `SAVE_PATH` against `checkpoint_path` |
| `invalid device ordinal` | Check `cuda:1` in `evaluation_evonet50.py` against the number of visible GPUs |
| `size mismatch` | Check the ResNet-12 / ResNet-50 backbone, `z_dim`, and checkpoint configuration |
| `too many values to unpack` | Update how `train_debug.py` unpacks the return values of `evolve_all()`, as described above |
| `OrderedDict` has no `.eval()` | Update the TIM experiment to construct the model and then load its `state_dict`, as described above |

Dependency installation, smoke tests, and full dataset training verify different parts of the workflow. Before a long experiment, check the dependencies, test data sampling, and make the configuration adjustments required by the selected script.
