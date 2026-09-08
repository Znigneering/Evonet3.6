# EvoNet for Python 3.6

Few-shot image classification with evolving sparse connections, Prototypical Networks, and genetic programming-based transductive boosting (TIM). Supports Omniglot and local OmniImage data.

The main backbone is ResNet-12. **ResNet-50 is still in experiments.**

## Installation

Use **64-bit Python 3.6.2 or a later 3.6 patch release**. Run commands from the repository root.

```bash
git clone https://github.com/Znigneering/Evonet3.6.git
cd Evonet3.6
conda create -n evonet36 python=3.6 -y
conda activate evonet36
python -m pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](requirements.txt) pins PyTorch `1.10.2`, torchvision `0.11.3`, NumPy `1.19.5`, Pillow `8.4.0`, Numba `0.53.1`, llvmlite `0.36.0`, tqdm `4.64.1`, and their required runtime dependencies.

<details>
<summary>Explicit CPU / CUDA 11.3 builds for Linux and Windows</summary>

Choose one of the following builds. CUDA requires a compatible NVIDIA GPU and driver.

CPU:

```bash
python -m pip install "torch==1.10.2+cpu" "torchvision==0.11.3+cpu" -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

CUDA 11.3:

```bash
python -m pip install "torch==1.10.2+cu113" "torchvision==0.11.3+cu113" -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

</details>

## Data

| Dataset | Official source | Download |
| --- | --- | --- |
| Omniglot | [Repository](https://github.com/brendenlake/omniglot) | [Background ZIP](https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip) / [Evaluation ZIP](https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip) |
| OmniImage (100 images per class) | [Repository](https://github.com/lfrati/OmnImage) | [OmnImage84_100.zip](https://www.uvm.edu/~lfrati/OmnImage84_100.zip) |

- **Omniglot:** downloaded automatically to `../data`. Training uses background classes; validation and evaluation use evaluation classes.
- **OmniImage:** extract the archive under `../data` to obtain `../data/OmnImage84_100/<class_name>/<image_file>`, or update `DATA_ROOT`. Supported formats: PNG, JPG, JPEG, and BMP. Classes are split 80/20 with seed `42`; validation and testing share the held-out classes.

Images are resized to 84 x 84 with three channels. For OmniImage, each class needs at least `k_shot + n_query` images, and each split must retain at least `n_way` valid classes. Omniglot samples with replacement when images are insufficient, so support/query overlap is possible.

## Training and evaluation

Run an entry point with `python <script.py>`. Configure episode counts, learning rates, data paths, and checkpoint paths inside the scripts.

| Dataset / backbone | Training | Evaluation |
| --- | --- | --- |
| Omniglot / ResNet-12 | [train_debug.py](train_debug.py) | [evaluation_evonet.py](evaluation_evonet.py) |
| Omniglot / ResNet-50 (in experiments) | [train_omniglot_evonet.py](train_omniglot_evonet.py) | Requires a matching Omniglot evaluation setup |
| OmniImage / ResNet-50 (in experiments) | [train_omniimage_evonet50.py](train_omniimage_evonet50.py) | [evaluation_evonet50.py](evaluation_evonet50.py) |

**Before running:**

- In `train_debug.py`, replace the three-value unpacking of `evolve_all()` with:

  ```python
  pruned, grown, _, _ = backbone.evolve_all(prune_threshold=PRUNE_THRESHOLD)
  ```

- `evaluation_evonet50.py` selects `cuda:1`; use `cuda:0` if only one GPU is visible.
- Match checkpoints to the backbone and `z_dim`. Omniglot ResNet-50 uses `z_dim=640`; the other training scripts use `64`. Weights are saved under `Baseline/` and are not included in the repository.

Evaluation concatenates support and query images for transductive batch normalization; validation during training processes them separately. The connection masks operate on dense tensors, so sparsity does not imply proportional runtime or memory savings.

## TIM boosting

[additive_boosting_tim.py](additive_boosting_tim.py) implements the TIM objective and evolutionary additive boosting. The objective uses query features without using query labels.

Run its synthetic-data smoke test without a dataset or checkpoint:

```bash
python additive_boosting_tim.py
```

[transductive_tim_omniglot.py](transductive_tim_omniglot.py) compares ProtoNet and TIM boosting on Omniglot. Its loader currently expects a full model, while training saves a wrapper `state_dict`: construct the matching backbone, remove the `backbone.` key prefix, and load the weights with `load_state_dict()` before running it.

Dependency installation, model forward passes and connection evolution, synthetic image sampling, and the TIM smoke test were verified on Windows x64 with Python 3.6.8 and PyTorch `1.10.2+cpu`. Full training and CUDA execution were not part of that check.

## Citation

If you use this work, please cite:

Zhilei Zhou and Malcolm I. Heywood. **Optimizing Transductive Few Shot Learning with Evolutionary Computation**.

```bibtex
@misc{zhou_transductive_evolution,
  author = {Zhou, Zhilei and Heywood, Malcolm I.},
  title  = {{Optimizing Transductive Few Shot Learning with Evolutionary Computation}}
}
```
