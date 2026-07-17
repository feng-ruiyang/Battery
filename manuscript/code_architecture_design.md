# BNN–Transformer 电池 SoH 概率轨迹模型代码架构与接口设计

版本日期：2026-07-14  
适用范围：Severson LFP 数据集预实验、VBLL + 显式退化动力学主方案、主要基线与消融实验

## 1. 代码目标

本项目代码需要支持从原始电池循环数据到论文实验结果的完整流程：

1. 解析和清洗电池循环数据，按 cell-level 划分训练、验证和测试集。
2. 训练 Transformer Encoder–Decoder，获得单循环 latent 表示。
3. 缓存每个循环的 latent 表示，供后续预测头、基线和消融共享。
4. 训练确定性 SoH 轨迹预测头，建立稳定基线。
5. 训练 VBLL 概率预测头，并加入显式退化动力学。
6. 输出 SoH 预测轨迹、概率云、RUL/EOL 分布和不确定性分解。
7. 统一评估点预测精度、概率校准、泛化能力和计算成本。

核心原则是先保证主方案清晰可复现，再逐步加入 MC-BNN、Deep Ensemble、Laplace 和 latent dynamics 等扩展。

## 2. 建议目录结构

```text
code/
  configs/                         # YAML 配置文件
    pretrain_encoder.yaml
    train_vbll_dynamics.yaml
    eval.yaml
  data/                            # 原始数据、处理后数据或软链接
    raw/
    processed/
    latent_cache/
  notebooks/                       # 数据检查和可视化，不放核心逻辑
  outputs/                         # 日志、checkpoint、预测结果和图表
  scripts/                         # 命令行入口脚本
    preprocess.py
    pretrain_encoder.py
    cache_latents.py
    train_predictor.py
    evaluate.py
    sample_trajectories.py
  src/
    battery_soh/
      data/
        datasets.py
        transforms.py
        splits.py
        datamodule.py
      models/
        encoder_decoder.py
        history_aggregator.py
        deterministic_head.py
        vbll.py
        degradation_dynamics.py
        soh_predictor.py
        baselines.py
      training/
        losses.py
        trainer.py
        callbacks.py
        checkpointing.py
      uncertainty/
        trajectory_distribution.py
        calibration.py
        rul_eol.py
        sampling.py
      evaluation/
        metrics.py
        evaluator.py
        plots.py
        reports.py
      utils/
        config.py
        seed.py
        logging.py
        device.py
  tests/
    test_splits.py
    test_dynamics.py
    test_distribution.py
    test_metrics.py
```

说明：

- `src/battery_soh/` 存放可复用 Python 包代码。
- `scripts/` 只负责读取配置、调用模块和保存结果，不写复杂模型逻辑。
- `notebooks/` 只用于探索性分析，不能作为正式实验依赖。
- `outputs/` 下按实验名和时间戳保存结果，避免覆盖。
- `latent_cache/` 是成本控制关键：不同预测头和消融共享同一组 Encoder 输出。

## 3. 数据接口设计

### 3.1 分层数据设计

当前阶段不直接冻结样本格式。不同电池数据集的字段、层级、采样频率和 SoH 标签定义可能不同，因此数据接口分为三层：

```text
RawRecord
  -> CanonicalCycle
  -> TrainingWindow
```

- `RawRecord`：贴近原始文件结构，由每个数据集自己的 parser 读取，不强行统一字段。
- `CanonicalCycle`：统一到 cell-cycle 层级，保留 `cell_id`、`cycle_index`、summary 字段和 cycle 内时间序列；字段允许 optional。
- `TrainingWindow`：模型真正使用的固定张量样本，由配置决定通道、窗口长度、预测范围和标签定义。

Severson LFP 的原始结构已经在 `manuscript/severson_data_structure_notes.md` 中单独记录。其他数据集下载小样本后，也应先写对应 audit，再决定共同的 `CanonicalCycle` 字段。

### 3.2 候选训练窗口格式

下面只是候选 `TrainingWindow` 格式，不是最终数据格式。真实实现应在三个数据集小样本审查后再冻结：

```python
sample = {
    "cell_id": str,
    "dataset_id": str,
    "batch_id": str | None,
    "cycle_index": int,
    "x_window": Tensor,        # [W, L, d_x]，原始循环信号窗口，可选
    "z_window": Tensor,        # [W, d_z]，缓存 latent 表示
    "u_window": Tensor,        # [W, d_u]，历史工况变量
    "u_future": Tensor,        # [H, d_u]，未来已知或假设工况变量
    "soh_history": Tensor,     # [W]
    "soh_future": Tensor,      # [H]
    "mask": Tensor,            # [H]，处理不同长度轨迹
}
```

第一阶段预训练使用 `x_window` 或单循环 `x_cycle`；第二阶段预测优先使用 `z_window`，避免重复运行 Encoder。

### 3.3 主要 Dataset

```python
class CycleReconstructionDataset(torch.utils.data.Dataset):
    """用于 Encoder–Decoder 预训练，返回单循环信号。"""

    def __getitem__(self, index) -> dict:
        return {
            "cell_id": str,
            "cycle_index": int,
            "x_cycle": Tensor,   # [L, d_x]
            "u_cycle": Tensor,   # [d_u]
        }
```

```python
class SOHTrajectoryDataset(torch.utils.data.Dataset):
    """用于 SoH 多步预测，返回历史窗口和未来轨迹。"""

    def __getitem__(self, index) -> dict:
        return sample
```

### 3.4 数据集 Parser 接口

每个数据集单独实现 parser，但输出统一的中间表示：

```python
class BaseBatteryParser:
    dataset_id: str

    def scan(self, raw_dir: Path) -> dict:
        """扫描原始文件，返回文件清单、元数据和可用字段。"""

    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        """将原始数据转换为 CanonicalCycle 表和元数据表。"""
```

Severson 第一版 parser 应支持：

- 读取作者 `.pkl` nested dictionary 或 `.mat` struct。
- 保留 `cycle_life`、`charge_policy`、`summary`、`cycles`。
- 从 summary 中提取 `cycle`、`QD`、`QC`、`IR`、`Tavg`、`Tmin`、`Tmax`、`chargetime`。
- 从 cycle 中提取 `I`、`V`、`T`、`Qc`、`Qd`、`Qdlin`、`Tdlin`、`dQdV`、`t`。
- 保存原论文 train/test/secondary-test split 和本项目 cell-level split。

### 3.5 划分规则

必须按 `cell_id` 划分训练、验证和测试集，禁止同一电池的循环跨集合出现。建议接口：

```python
def make_cell_level_split(
    cell_ids: list[str],
    split_name: str,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """返回 train/val/test 的 cell_id 列表。"""
```

所有实验需要把划分结果保存为 JSON，保证后续基线和消融使用完全相同的数据划分。

## 4. 模型模块设计

### 4.1 Transformer Encoder–Decoder

```python
class CycleEncoder(nn.Module):
    def forward(self, x_cycle: Tensor, u_cycle: Tensor | None = None) -> Tensor:
        """输入 [B, L, d_x]，输出 [B, d_z]。"""
```

```python
class CycleDecoder(nn.Module):
    def forward(self, z: Tensor, x_prefix: Tensor | None = None) -> dict:
        """自回归或 teacher-forcing 重构单循环信号。"""
```

```python
class CycleAutoencoder(nn.Module):
    def forward(self, x_cycle: Tensor, u_cycle: Tensor | None = None) -> dict:
        return {
            "z": Tensor,              # [B, d_z]
            "x_recon": Tensor,        # [B, L, d_x]
            "reconstruction_loss": Tensor,
        }
```

### 4.2 历史聚合器

```python
class HistoryAggregator(nn.Module):
    def forward(
        self,
        z_window: Tensor,     # [B, W, d_z]
        u_window: Tensor,     # [B, W, d_u]
        soh_history: Tensor | None = None,
    ) -> Tensor:
        """输出历史状态 h_t: [B, d_h]。"""
```

第一版建议使用小型 Transformer Encoder 或 GRU。若数据量较小，优先控制模型规模，避免历史聚合器过拟合。

### 4.3 VBLL 最后一层

```python
class VBLLLinear(nn.Module):
    def forward(self, features: Tensor) -> dict:
        return {
            "theta_mean": Tensor,      # [B, 1 + K]
            "theta_var": Tensor,       # [B, 1 + K]
            "kl": Tensor,
        }
```

其中 `theta = [s_t, c_1, ..., c_K]`，`s_t` 为当前 SoH，`c_k` 为退化 basis 系数。第一版采用 mean-field Gaussian posterior，避免完整协方差带来的计算和数值成本。

### 4.4 显式退化动力学

```python
class DegradationDynamics(nn.Module):
    def forward(
        self,
        theta_mean: Tensor,    # [B, 1 + K]
        theta_var: Tensor,     # [B, 1 + K]
    ) -> dict:
        return {
            "y_mean": Tensor,          # [B, H]
            "y_cov": Tensor,           # [B, H, H]
            "rate_mean": Tensor,       # [B, H]
            "regularization": dict,
        }
```

该模块内部构造 basis matrix `B` 和轨迹传播矩阵 `G`：

```python
class BasisFactory:
    @staticmethod
    def make_basis(kind: str, horizon: int, num_basis: int) -> Tensor:
        """支持 bspline、piecewise_linear、polynomial 等 basis。"""
```

动力学正则项包括：

- `negative_rate_penalty`：惩罚明显为负的退化速率。
- `smoothness_penalty`：惩罚退化率的二阶差分振荡。

### 4.5 主预测模型

```python
class SOHProbabilisticPredictor(nn.Module):
    def forward(self, batch: dict) -> dict:
        return {
            "y_mean": Tensor,          # [B, H]
            "y_cov": Tensor,           # [B, H, H]
            "theta_mean": Tensor,      # [B, 1 + K]
            "theta_var": Tensor,       # [B, 1 + K]
            "rate_mean": Tensor,       # [B, H]
            "kl": Tensor,
            "regularization": dict,
        }
```

确定性基线可复用相同的 `HistoryAggregator` 和 `DegradationDynamics`，只把 `VBLLLinear` 替换为普通 `nn.Linear`。

## 5. 损失函数接口

```python
def gaussian_trajectory_nll(
    y_true: Tensor,       # [B, H]
    y_mean: Tensor,       # [B, H]
    y_cov: Tensor,        # [B, H, H]
    mask: Tensor | None = None,
    jitter: float = 1e-5,
) -> Tensor:
    """多元高斯负对数似然。"""
```

```python
def total_loss(outputs: dict, batch: dict, cfg: dict) -> dict:
    return {
        "loss": Tensor,
        "nll": Tensor,
        "kl": Tensor,
        "negative_rate": Tensor,
        "smoothness": Tensor,
        "reconstruction": Tensor | None,
    }
```

`y_cov` 必须加入最小对角 jitter，确保 Cholesky 分解稳定。若早期训练不稳定，可先使用对角协方差，再切换到由 `G diag(v_theta) G^T` 形成的完整轨迹协方差。

## 6. 不确定性、RUL 与 EOL 接口

```python
def sample_trajectories(
    y_mean: Tensor,
    y_cov: Tensor,
    num_samples: int,
) -> Tensor:
    """从联合高斯轨迹分布采样，输出 [B, M, H]。"""
```

```python
def estimate_eol_distribution(
    trajectories: Tensor,       # [B, M, H]
    current_cycle: Tensor,      # [B]
    threshold: float = 0.8,
) -> dict:
    return {
        "eol_samples": Tensor,  # [B, M]
        "rul_samples": Tensor,  # [B, M]
        "eol_median": Tensor,
        "rul_interval": Tensor,
    }
```

```python
def decompose_uncertainty(outputs: dict) -> dict:
    return {
        "epistemic": Tensor,    # 来自 theta posterior 的轨迹方差
        "aleatoric": Tensor,    # 来自观测噪声/残差项
        "total": Tensor,
    }
```

主方案的概率云生成只需要高斯采样和矩阵乘法，不需要重复运行 Transformer。

## 7. 评估指标接口

```python
def point_metrics(y_true: Tensor, y_pred: Tensor, mask: Tensor | None = None) -> dict:
    return {
        "mae": float,
        "rmse": float,
        "horizon_mae": list[float],
        "horizon_rmse": list[float],
    }
```

```python
def probabilistic_metrics(
    y_true: Tensor,
    y_mean: Tensor,
    y_cov: Tensor,
    samples: Tensor | None = None,
) -> dict:
    return {
        "nll": float,
        "crps": float,
        "coverage_50": float,
        "coverage_80": float,
        "coverage_90": float,
        "coverage_95": float,
        "interval_width_90": float,
        "calibration_error": float,
    }
```

所有指标应同时报告总体平均值、按 horizon 的曲线，以及按 cell 聚合后的置信区间。

## 8. 配置文件设计

建议使用 YAML 管理实验配置：

```yaml
experiment:
  name: vbll_dynamics_severson
  seed: 42
  output_dir: outputs/vbll_dynamics_severson

data:
  dataset: severson_lfp
  raw_dir: data/raw/severson
  processed_dir: data/processed/severson
  latent_cache_dir: data/latent_cache/severson
  split_file: data/processed/severson/splits/seed_42.json
  window: 30
  horizon: 80
  sequence_length: 1000

model:
  latent_dim: 128
  history_dim: 256
  num_basis: 8
  basis: bspline
  covariance: full_from_theta
  aleatoric: diagonal

training:
  batch_size: 64
  max_epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-5
  beta_kl: 1.0e-4
  lambda_negative: 1.0
  lambda_smooth: 0.1
  precision: bf16-mixed
  early_stopping_patience: 20

logging:
  backend: wandb              # 可选：wandb、tensorboard、csv
  project: battery-soh-bnn
  entity: null                # 个人账号可留空，团队账号填写 team/entity 名称
  group: severson_vbll
  tags:
    - severson
    - vbll
    - degradation-dynamics
  log_model: false            # checkpoint 默认先保存在本地，重要版本再上传
  offline: false              # 无网络或集群环境可设为 true
```

## 9. 命令行入口

### 9.0 Anaconda 环境配置

项目环境统一使用 Anaconda 或 Miniconda 管理。建议在 `code/` 下维护一个 `environment.yaml`：

```yaml
name: battery-soh
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch
  - pytorch-cuda=12.1
  - numpy
  - scipy
  - pandas
  - pyarrow
  - scikit-learn
  - matplotlib
  - seaborn
  - pyyaml
  - tqdm
  - pytest
  - pip
  - pip:
      - lightning
      - einops
      - wandb
      - ruff
```

创建和激活环境：

```bash
conda env create -f environment.yaml
conda activate battery-soh
```

如果租用平台 CUDA 版本与 `pytorch-cuda=12.1` 不匹配，应以平台镜像的驱动和 PyTorch 官方安装命令为准，保持项目代码只依赖 `environment.yaml` 记录最终可复现环境。

### 9.1 数据预处理

```bash
python scripts/preprocess.py --config configs/preprocess_severson.yaml
```

输出：

- 统一格式的循环信号文件；
- SoH 标签文件；
- cell-level 划分 JSON；
- 数据质量检查报告。

### 9.2 预训练 Encoder–Decoder

```bash
python scripts/pretrain_encoder.py --config configs/pretrain_encoder.yaml
```

输出：

- Encoder–Decoder checkpoint；
- 重构误差曲线；
- 单循环 latent 可视化。

### 9.3 缓存 latent 表示

```bash
python scripts/cache_latents.py \
  --config configs/pretrain_encoder.yaml \
  --checkpoint outputs/pretrain/checkpoints/best.ckpt
```

输出：

- 每个 `cell_id` 和 `cycle_index` 对应的 `z_t`；
- latent cache 的索引文件。

### 9.4 训练 SoH 预测模型

```bash
python scripts/train_predictor.py --config configs/train_vbll_dynamics.yaml
```

输出：

- 模型 checkpoint；
- 训练、验证 NLL 和 MAE 曲线；
- 验证集覆盖率与 calibration report。

若启用 wandb，训练脚本应在启动时读取 `logging` 配置并调用 `wandb.init()`，自动记录配置、loss、指标、学习率、显存和关键图表。

### 9.5 评估和画图

```bash
python scripts/evaluate.py \
  --config configs/eval.yaml \
  --checkpoint outputs/vbll_dynamics_severson/checkpoints/best.ckpt
```

输出：

- `metrics.json`；
- `predictions.parquet`；
- SoH 轨迹图、概率云图、可靠性图；
- RUL/EOL 分布图；
- 推理时间和显存统计。

### 9.6 wandb API 配置

wandb API key 不应写入 YAML、Python 文件或 git。推荐使用环境变量或命令行登录：

```bash
conda activate battery-soh
wandb login
```

或在服务器上使用：

```bash
export WANDB_API_KEY="your_api_key_here"
```

训练入口建议支持以下行为：

```python
if cfg.logging.backend == "wandb":
    import wandb

    mode = "offline" if cfg.logging.offline else "online"
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        group=cfg.logging.group,
        tags=cfg.logging.tags,
        name=cfg.experiment.name,
        config=cfg,
        mode=mode,
    )
```

每个 epoch 至少记录：

- `train/loss`、`train/nll`、`train/kl`；
- `val/mae`、`val/rmse`、`val/nll`；
- `val/coverage_50`、`val/coverage_80`、`val/coverage_90`、`val/coverage_95`；
- `val/interval_width_90`、`val/calibration_error`；
- `system/gpu_memory_gb`、`system/epoch_time_sec`；
- 若生成图表，则上传 SoH 轨迹图、概率云图、reliability diagram 和 RUL/EOL 分布图。

多随机种子实验建议用同一个 `group`，用不同 `name` 标识 seed 和配置，例如：

```text
vbll_dynamics_seed_42
vbll_dynamics_seed_43
vbll_dynamics_seed_44
```

无网络或需要离线训练时：

```bash
export WANDB_MODE=offline
```

训练结束后再同步：

```bash
wandb sync wandb/offline-run-*
```

## 10. 预计使用的库

### 10.1 核心训练

- `Anaconda` 或 `Miniconda`：统一管理 Python、CUDA 版 PyTorch 和实验依赖。
- `python >= 3.11`：由 conda 环境固定版本。
- `torch`：深度学习主框架，优先通过 conda 安装与 CUDA 匹配的版本。
- `torchvision`：可选，用于通用 transform 或工具函数。
- `lightning` 或原生 PyTorch Trainer：建议优先使用 Lightning 管理训练循环、日志和 checkpoint；若调试复杂概率损失不方便，可退回原生 PyTorch。
- `einops`：简化张量维度重排。
- `numpy`、`scipy`：数值计算、B-spline basis、统计分布。
- `pandas`、`pyarrow`：表格数据和 parquet 文件读写。
- `scikit-learn`：数据划分、基础指标、标准化。

### 10.2 概率与校准

- `torch.distributions`：高斯分布、采样和 log probability。
- `laplace-torch`：Last-layer Laplace 基线，条件启用。
- `netcal` 或自定义实现：校准曲线和 calibration error，优先先自实现核心指标，减少依赖。

### 10.3 配置、日志与实验管理

- `PyYAML` 或 `omegaconf`：配置文件读取。
- `tensorboard`：训练曲线记录。
- `wandb`：实验追踪主工具；通过 `WANDB_API_KEY` 或 `wandb login` 配置 API，不把 key 写入仓库。
- `tqdm`：进度条。
- `rich`：更清晰的命令行日志，可选。

### 10.4 可视化与报告

- `matplotlib`：论文图和基础可视化。
- `seaborn`：统计图美化，可选。
- `plotly`：交互式轨迹检查，可选，不作为论文图依赖。

### 10.5 工程质量

- `pytest`：单元测试。
- `ruff`：代码格式和静态检查。
- `mypy`：类型检查，可选。
- `pre-commit`：提交前检查，可选。

## 11. 优先实现顺序

### 第一阶段：可跑通

1. 数据解析、cell-level split 和 `SOHTrajectoryDataset`。
2. 确定性历史聚合器和确定性动力学头。
3. MAE/RMSE/NLL 基础指标。
4. 单随机种子端到端训练和评估。

### 第二阶段：主方案

1. `VBLLLinear`。
2. `DegradationDynamics` 中的 `B`、`G` 和解析协方差传播。
3. 多元高斯 NLL、KL、退化速率正则和平滑正则。
4. 覆盖率、区间宽度、calibration error 和概率云采样。

### 第三阶段：生成式预训练

1. `CycleEncoder` 和 `CycleDecoder`。
2. Encoder–Decoder 重构训练。
3. latent cache 生成与复用。
4. 冻结 Encoder 与部分解冻 Encoder 的消融。

### 第四阶段：基线与论文实验

1. MC Dropout、Last-layer Laplace、Bayes by Backprop。
2. Deep Ensemble。
3. CNN–PatchTST 或其他强确定性基线。
4. 多随机种子、多划分和跨工况测试。

## 12. 最小测试计划

必须优先写以下单元测试，避免后期结果不可追踪：

- `test_splits.py`：确认 train/val/test 没有重复 `cell_id`。
- `test_dynamics.py`：确认 `theta -> y_mean/y_cov` 维度正确，`y_cov` 正定。
- `test_distribution.py`：确认 NLL、采样和区间覆盖率接口正常。
- `test_metrics.py`：确认 MAE、RMSE、coverage 与 mask 逻辑正确。

每次长训练前应先运行：

```bash
pytest tests
```

## 13. 主要工程风险

| 风险 | 影响 | 应对 |
|---|---|---|
| cell-level 泄漏 | 指标虚高，论文结果无效 | split 文件持久化，测试强制检查 |
| 多元协方差不正定 | NLL 训练崩溃 | 加 jitter，先从对角协方差 warm-up |
| Encoder 重复训练 | GPU 成本暴涨 | 按 split 和 seed 缓存 latent |
| 自回归重构过慢 | 预训练时间过长 | 先用短序列或非自回归重构验证 |
| 不确定性低估 | 区间覆盖率不足 | 比较 VBLL、MC-BNN、Ensemble 和 Laplace |
| notebook 逻辑外溢 | 实验不可复现 | 核心逻辑必须进入 `src/battery_soh/` |

## 14. 第一版完成标准

第一版代码完成后，应能用一条训练命令完成以下流程：

1. 读取处理后的 Severson 数据。
2. 使用固定 cell-level split。
3. 训练一个确定性动力学头和一个 VBLL + 显式动力学模型。
4. 在测试集输出 MAE、RMSE、NLL、覆盖率和区间宽度。
5. 为若干电池生成 SoH 概率轨迹图和 RUL/EOL 分布图。
6. 保存所有配置、checkpoint、预测结果和指标文件。

达到该标准后，再启动完整基线、消融和多随机种子实验。
