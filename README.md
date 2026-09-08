# EvoNet for Python 3.6

基于 PyTorch 的小样本图像分类实验代码，包含可演化连接的 ResNet-12 / ResNet-50、原型网络（Prototypical Networks），以及使用 TIM（Transductive Information Maximization）目标的遗传编程加性 Boosting。

本仓库面向 **64 位 Python 3.6** 环境。依赖版本固定在 [requirements.txt](requirements.txt) 中；PyTorch 要求 Python **3.6.2 或更高的 3.6 补丁版本**。下文命令均从仓库根目录执行。

## 实现内容

- **可演化网络连接**：卷积层与线性层通过掩码控制有效连接，按权重幅值剪枝，并利用梯度信号选择重新激活的位置。`Evonet.py` 和 `Evonet50.py` 使用直通估计器为非活跃连接传递梯度。
- **小样本分类**：每个 episode 从 support set 计算类别原型，使用 query 特征到原型的平方欧氏距离进行分类。
- **两种骨干网络**：`Evonet.py` 实现 ResNet-12；`Evonet50.py` 实现 ResNet-50，并使用基于当前 batch 统计量的 `EpisodeBatchNorm2d`。
- **两类图像数据**：支持自动下载的 Omniglot，以及按类别文件夹组织的本地 `OmnImage84_100` 数据。
- **TIM Boosting**：`additive_boosting_tim.py` 使用遗传编程搜索加性特征变换，并通过 Numba 加速 TIM 损失计算。该损失使用 support 标签与 query 特征，不使用 query 标签计算优化目标。

`density` 表示初始化时的有效连接比例，不保证演化后保持不变。掩码应用在常规稠密张量上，不能直接将有效连接比例视为同等比例的运行时间或显存节省。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `Evonet.py` | 可演化 ResNet-12、特征投影层与 Omniglot episode 采样器 |
| `Evonet50.py` | 可演化 ResNet-50、Episode BatchNorm 与 Omniglot episode 采样器 |
| `train_debug.py` | 在 Omniglot 上训练 ResNet-12；运行前需调整下文所述的返回值解包 |
| `train_omniglot_evonet.py` | 在 Omniglot 上训练 **ResNet-50**，投影维度为 640 |
| `train_omniimage_evonet50.py` | 在本地 OmniImage 数据上训练 ResNet-50，投影维度为 64 |
| `omni_image_sampler.py` | 本地图像类别划分、预处理与 episode 采样 |
| `evaluation_evonet.py` | 在 Omniglot 上评估 ResNet-12 的 ProtoNet 权重 |
| `evaluation_evonet50.py` | 在 OmniImage 上评估 ResNet-50 的 ProtoNet 权重 |
| `additive_boosting_tim.py` | TIM 损失、遗传编程 Boosting 与随机数据自检入口 |
| `transductive_tim_omniglot.py` | Omniglot 上的 ProtoNet / TIM Boosting 对比实验；需适配权重加载方式 |

仓库没有统一的命令行参数解析器。学习率、episode 数、数据目录、模型维度与保存路径均在对应脚本中配置，不能通过 `--epochs` 等参数覆盖。

## 安装

### 1. 获取代码并创建 Python 3.6 环境

```bash
git clone https://github.com/Znigneering/Evonet3.6.git
cd Evonet3.6
conda create -n evonet36 python=3.6 -y
conda activate evonet36
python --version
```

如果已安装 Python 3.6，也可以使用 `venv`。Linux / macOS：

```bash
python3.6 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
py -3.6 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 固定安装工具版本

在刚创建并激活的环境中执行：

```bash
python -m pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"
```

这些版本支持 Python 3.6，避免安装工具升级到要求更新 Python 的版本。后续命令统一使用 `python -m pip`，确保依赖安装到当前解释器所在环境。

### 3. 安装项目依赖

使用 PyPI 提供的默认构建：

```bash
python -m pip install -r requirements.txt
```

如果需要明确选择 CPU 或 CUDA 构建，在新环境中选择下面一种方式。先安装匹配的 `torch` / `torchvision`，再安装完整依赖。

**仅 CPU，Linux / Windows：**

```bash
python -m pip install "torch==1.10.2+cpu" "torchvision==0.11.3+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install -r requirements.txt
```

**CUDA 11.3，Linux / Windows：**

```bash
python -m pip install "torch==1.10.2+cu113" "torchvision==0.11.3+cu113" -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install -r requirements.txt
```

CUDA 构建需要支持该构建的 NVIDIA GPU 和驱动。`+cpu` 与 `+cu113` 都满足 requirements 中不带后缀的版本约束。Intel macOS 可使用默认安装方式；上述 `+cpu` / `+cu113` 命令面向 Linux / Windows，本依赖组合未覆盖原生 Apple Silicon 环境。可用构建见 PyTorch 官方的 [CPU wheel 列表](https://download.pytorch.org/whl/cpu/torch_stable.html) 和 [CUDA 11.3 wheel 列表](https://download.pytorch.org/whl/cu113/torch_stable.html)。

### 4. 检查安装

```bash
python -m pip check
python -c "import sys, torch, torchvision, numpy, PIL, tqdm, numba, llvmlite; print('Python:', sys.version); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('Pillow:', PIL.__version__); print('tqdm:', tqdm.__version__); print('Numba:', numba.__version__); print('llvmlite:', llvmlite.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
```

不需要数据集或模型权重的 TIM / Boosting 自检：

```bash
python additive_boosting_tim.py
```

该脚本生成随机数据，计算 TIM 损失并运行一个小规模 Boosting 实验；正常完成时打印 `Fit complete.`。首次调用包含 Numba 编译时间，不能将其直接当作稳定运行耗时。

已在 Windows x64、CPython 3.6.8、PyTorch `1.10.2+cpu` / torchvision `0.11.3+cpu` 的隔离环境中验证依赖安装与 `pip check`、全部 Python 源码编译、两种骨干网络的随机输入前向计算与连接演化、合成图像的 episode 采样，以及上述 TIM / Boosting 自检。该验证不包含 CUDA 运行或真实数据集上的完整训练与准确率复现。

### 依赖版本

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| PyTorch | `1.10.2` | 网络、自动微分与优化器 |
| torchvision | `0.11.3` | Omniglot 数据集与图像变换，与 PyTorch 版本配套 |
| NumPy | `1.19.5` | episode 采样与 Boosting 数值计算 |
| Pillow | `8.4.0` | 图像读取，对应代码中的 `PIL` |
| tqdm | `4.64.1` | 训练与评估进度条 |
| Numba / llvmlite | `0.53.1` / `0.36.0` | TIM 损失的 JIT 编译 |

requirements 同时固定 `setuptools`、`typing-extensions`、Python 3.6 所需的 `dataclasses` / `importlib-resources` / `zipp`，以及 Windows 下的 `colorama`，覆盖运行时的间接依赖。版本信息可查阅 [PyTorch 1.10.2](https://pypi.org/project/torch/1.10.2/)、[torchvision 0.11.3](https://pypi.org/project/torchvision/0.11.3/)、[Numba 0.53.1](https://pypi.org/project/numba/0.53.1/) 与 [tqdm 4.64.1](https://pypi.org/project/tqdm/4.64.1/) 的发布记录。

## 数据准备

### Omniglot

`OmniglotBoosterTaskSampler` 使用 `torchvision.datasets.Omniglot`，首次运行设置了 `download=True`，需要联网。数据根目录为相对于当前工作目录的 `../data`。

- 训练使用 `background=True`，验证和评估使用 `background=False`。
- 图像缩放到 84 × 84，转换为张量、反转像素值，再将单通道复制为三通道。
- 一个 episode 返回 support 图像与标签、query 图像与标签；标签在 episode 内重新编号为 `0` 到 `n_way - 1`。
- 每类需要 `k_shot + n_query` 张图像；Omniglot 采样器在数量不足时使用有放回采样，可能使 support 和 query 出现重复图像。

### OmniImage / `OmnImage84_100`

该数据集需要自行准备，本仓库不提供下载器或数据文件。训练与评估脚本默认查找仓库上一级的 `data/OmnImage84_100`，目录名按代码中的拼写使用：

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

根目录的每个直接子文件夹代表一个类别；图片直接放在类别文件夹内，支持 `.png`、`.jpg`、`.jpeg` 和 `.bmp`。不需要额外的 `train/`、`val/`、`test/` 文件夹层级。

采样器先排序类别名，再使用固定种子 `42` 打乱，默认按类别划分 80% 训练、20% 测试；`val` 与 `test` 使用同一批保留类别，并非独立的三个划分。每类至少要有 `k_shot + n_query` 张图片，否则会被过滤；过滤后所选划分至少需要 `n_way` 个有效类别。图像转换为 RGB，缩放为 84 × 84，再转换到 `[0, 1]` 张量。

可以先验证数据读取：

```bash
python omni_image_sampler.py
```

该自检默认采样 5-way、1-shot、每类 5 个 query，因此每类至少需要 6 张图片。训练和评估脚本通过 `DATA_ROOT` 修改数据位置；采样器自检使用其默认相对路径。

## 训练与评估

`n_way` 是每个 episode 的类别数，`k_shot` 是每类 support 样本数，`n_query` 是每类 query 样本数。以下为源码中的默认配置；显存不足时先降低类别数或每类 query 数。

| 训练入口 | 数据 / 骨干 | 训练 episode | 训练 way / shot / query | 验证 way / shot / query | `z_dim` / 初始 `density` |
| --- | --- | --- | --- | --- | --- |
| `train_debug.py` | Omniglot / ResNet-12 | 5,000 | 60 / 1 / 15 | 5 / 1 / 15 | 64 / 0.5 |
| `train_omniglot_evonet.py` | Omniglot / ResNet-50 | 10,000 | 80 / 1 / 1 | 20 / 1 / 1 | 640 / 1.0 |
| `train_omniimage_evonet50.py` | OmniImage / ResNet-50 | 100,000 | 20 / 1 / 1 | 5 / 1 / 1 | 64 / 1.0 |

学习率均为 `1e-3`。剪枝阈值与演化间隔分别为 `0.001` / 500、`0.01` / 50、`0.001` / 1,000。要先检查完整流程，可在脚本中降低 `N_TRAIN_EPISODES`、验证次数与采样规模。

### OmniImage：ResNet-50

准备数据后训练：

```bash
python train_omniimage_evonet50.py
```

权重保存在：

```text
Baseline/evo_omniimage/evonet50_omniimage_100k-epoch_1e-3-lr_1000-evo-interval.pth
```

评估前检查 `evaluation_evonet50.py` 的 `DEVICE`。当前代码在 CUDA 可用时固定使用 `cuda:1`，即第二张可见 GPU；只有一张 GPU 时，应改为：

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

随后执行：

```bash
python evaluation_evonet50.py
```

默认评估 600 个 20-way、1-shot、每类 1 个 query 的 episode。数据的测试划分需要至少 20 个有效类别。

### Omniglot：ResNet-12

**先调整 `train_debug.py` 的演化返回值解包。** `Evonet.py` 中的 `evolve_all()` 返回四个值，但该训练脚本当前只接收三个；首次演化（默认第 500 个 episode）会触发 `ValueError: too many values to unpack`。将对应语句改为：

```python
pruned, grown, _, _ = backbone.evolve_all(prune_threshold=PRUNE_THRESHOLD)
```

完成调整后运行：

```bash
python train_debug.py
python evaluation_evonet.py
```

训练脚本在验证准确率刷新最佳值时保存 `Baseline/evo_omniglot/evonet_test.pth`；评估脚本读取同一路径，默认运行 600 个 20-way、1-shot、每类 1 个 query 的 episode。

### Omniglot：ResNet-50

```bash
python train_omniglot_evonet.py
```

虽然文件名没有 `50`，它实际从 `Evonet50.py` 导入网络，使用 `z_dim=640`，并在训练结束后保存 `Baseline/evo_omniglot/evonet50_test.pth`。

当前仓库没有与这一路训练直接配套的评估入口：`evaluation_evonet.py` 使用 ResNet-12，`evaluation_evonet50.py` 使用 OmniImage 数据和 `z_dim=64`。评估此权重时需要同时匹配 **ResNet-50、Omniglot 采样器、`z_dim=640` 和权重路径**，仅替换文件名不足以完成适配。

### 权重与评估协议

训练脚本保存的是 `PrototypicalNetworks` 包装模型的 `state_dict`，参数键带有 `backbone.` 前缀。加载时必须匹配骨干类型与投影维度；`strict=False` 不能消除同名参数的形状不匹配。

两个 `evaluation_evonet*.py` 脚本将 support 和 query 拼接后共同提取特征，并使用当前 batch 的归一化统计量。这属于使用 query 分布信息的传导式评估；训练中的验证则分别处理 support 和 query，比较准确率时应同时记录这些设置。

`Baseline/` 下的权重由运行脚本生成，已被 `.gitignore` 排除；仓库没有附带预训练权重。各脚本会重复写入固定的 `SAVE_PATH`，需要保留多次实验结果时应先设置不同路径。

## TIM Boosting 对比实验

`transductive_tim_omniglot.py` 默认在 Omniglot 上对比原型分类和 TIM Boosting，设置为 10 个 5-way、5-shot、每类 15 个 query 的 episode，每个 episode 单独拟合一个 Boosting 模型。

**当前加载逻辑需要先适配。** 脚本将 `torch.load()` 的返回值当作完整网络直接调用 `.eval()` 与 `.encoder()`，而本仓库训练脚本保存的是 `state_dict`。直接使用 `train_debug.py` 生成的权重会遇到 `OrderedDict` 没有 `.eval()` 的错误。

对于上文 ResNet-12 的 `evonet_test.pth`，可将 `main()` 中从 `print("Loading Baseline Model...")` 到 `model.eval()` 的加载部分替换为以下代码，恢复实际的特征网络：

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

完成适配并准备好对应权重后执行：

```bash
python transductive_tim_omniglot.py
```

该实验输出 ProtoNet 准确率、Boosting 准确率及两者差值；结果取决于训练权重、采样任务与实验配置，仓库不附带可复现的准确率基准表。

## 常见问题

| 现象 | 检查方式 |
| --- | --- |
| `No matching distribution found` | 确认当前解释器为 64 位 Python 3.6.2+、pip 为 21.3.1，且使用本 README 对应平台的安装命令 |
| `No module named PIL` | 安装包名为 `Pillow`，已包含在 requirements 中 |
| `Dataset root not found` / 有效类别不足 | 检查 `DATA_ROOT`、类别文件夹层级、每类图片数和划分后的 `n_way` |
| `Checkpoint not found` | 先训练对应模型，并核对 `SAVE_PATH` 与 `checkpoint_path` |
| `invalid device ordinal` | 检查 `evaluation_evonet50.py` 中的 `cuda:1` 与实际可见 GPU 数量 |
| `size mismatch` | 检查 ResNet-12 / ResNet-50、`z_dim` 和训练权重是否匹配 |
| `too many values to unpack` | 按上文调整 `train_debug.py` 对 `evolve_all()` 返回值的接收方式 |
| `OrderedDict` 没有 `.eval()` | 按上文将 TIM 实验改为构建模型后加载 `state_dict` |

依赖安装、自检和完整数据集训练是不同层面的验证。开始长时间实验前，先完成依赖检查、数据采样检查以及与目标脚本对应的配置调整。
