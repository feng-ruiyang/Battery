# Severson LFP 数据结构初步笔记

来源仓库：`rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation`  
本地副本：`code/external/severson_processing/`  
记录日期：2026-07-14

## 1. 结论

作者仓库只提供数据处理代码，不提供建模代码。README 明确说明：

- MATLAB `.mat` 文件中，数据存储为 `struct`。
- Python `.pkl` 文件中，数据存储为 nested dictionaries。
- 每个电池 cell 的数据分为三类：`descriptors`、`summary`、`cycle`。

因此，目前不应把训练样本格式直接定死。更稳妥的设计是：

```text
dataset-specific raw file
  -> RawRecord
  -> CanonicalCycle
  -> TrainingWindow
```

其中 Severson parser 只负责把作者原始结构转换为项目内部的 `CanonicalCycle`，后续模型只依赖统一后的中间格式。

## 2. 原始文件与批次

`LoadData.m` 中使用三个主要 batch：

```matlab
2017-05-12_batchdata_updated_struct_errorcorrect
2017-06-30_batchdata_updated_struct_errorcorrect
2018-04-12_batchdata_updated_struct_errorcorrect
```

处理逻辑：

1. batch1 直接载入。
2. batch2 中部分电池是 batch1 的延续，需要追加到 batch1 前几个 cell。
3. batch2 的延续 cell 随后从 batch2 中删除。
4. batch3 删除一个数据采集异常 channel。
5. batch3 删除末端容量仍高于 0.885 的 cell。
6. batch3 删除 noisy Batch 8 batteries。
7. 可选删除 batch1 中未完成测试的电池。

这说明 cell-level ID 和 batch/cell 映射必须在预处理阶段显式保存，不能只依赖数组下标。

## 3. Cell 级字段

Python notebook 中读取到的 batch 顶层字段包括：

```text
barcode
channel_id
cycle_life
cycles
policy
policy_readable
summary
```

转换为 Python dict 后，单个 cell 结构近似为：

```python
cell_dict = {
    "cycle_life": cl,
    "charge_policy": policy,
    "summary": summary,
    "cycles": cycle_dict,
}
```

README 提到 descriptors 包括：

```text
charging policy
cycle life
barcode
channel
```

但 barcode 和 channel 当前不一定出现在 `.pkl` 文件中。

## 4. Summary 字段

`summary` 是按 cycle 聚合的低频数据。作者 notebook 中转换后的字段为：

```python
summary = {
    "IR": summary_IR,
    "QC": summary_QC,
    "QD": summary_QD,
    "Tavg": summary_TA,
    "Tmin": summary_TM,
    "Tmax": summary_TX,
    "chargetime": summary_CT,
    "cycle": summary_CY,
}
```

对应含义：

| 字段 | 含义 |
|---|---|
| `cycle` | 循环编号 |
| `QD` / `QDischarge` | 放电容量 |
| `QC` / `QCharge` | 充电容量 |
| `IR` | 内阻 |
| `Tmax` | 最高温度 |
| `Tavg` | 平均温度 |
| `Tmin` | 最低温度 |
| `chargetime` | 充电时间 |

SoH 可由容量归一化得到。作者寿命标签使用 0.88 容量阈值：

```matlab
if batch_combined(i).summary.QDischarge(end) < 0.88
    bat_label(i) = find(batch_combined(i).summary.QDischarge < 0.88,1);
else
    bat_label(i) = size(batch_combined(i).cycles,2) + 1;
end
```

本项目 proposal 中若使用 80% EOL 阈值，需要明确与原论文 88% cycle life 标签不同。

## 5. Cycle 字段

`cycle` 是单循环内部时间序列数据。作者 notebook 中每个 cycle 转换为：

```python
cd = {
    "I": I,
    "Qc": Qc,
    "Qd": Qd,
    "Qdlin": Qdlin,
    "T": T,
    "Tdlin": Tdlin,
    "V": V,
    "dQdV": dQdV,
    "t": t,
}
cycle_dict[str(j)] = cd
```

字段含义：

| 字段 | 含义 |
|---|---|
| `t` | cycle 内时间 |
| `I` | 电流 |
| `V` | 电压 |
| `T` | 温度 |
| `Qc` | 充电容量 |
| `Qd` | 放电容量 |
| `Qdlin` | 线性插值后的放电容量 |
| `Tdlin` | 线性插值后的温度 |
| `dQdV` | 放电微分容量曲线 |

这对本项目很重要：如果原始 cycle 长度不一致，可以优先考虑使用作者已经给出的 `Qdlin`、`Tdlin` 和统一电压网格上的 `dQdV`，再决定是否重新对 `V/I/T/Q` 做 resampling。

## 6. 训练/测试划分

`LoadData.m` 中给出原论文划分方式：

```matlab
test_ind = [1:2:(numBat1+numBat2),84];
train_ind = 1:(numBat1+numBat2);
train_ind(test_ind) = [];
secondary_test_ind = numBat-numBat3+1:numBat;
```

含义：

- batch1 + batch2 用于主训练/测试划分。
- batch3 作为 secondary test。
- 如果本项目要与原论文比较，需要保留该划分。
- 如果本项目做 SoH 轨迹预测，也应额外设计 cell-level random split 和跨 batch split。

## 7. 对本项目代码架构的影响

### 7.1 不能直接固定 `sample` 字典

不同数据集可能没有 `Qdlin`、`Tdlin`、`dQdV` 或 `IR`。因此 `code_architecture_design.md` 中的样本格式应标为候选格式，而不是最终格式。

### 7.2 建议定义三层数据对象

```python
RawRecord
CanonicalCycle
TrainingWindow
```

其中：

- `RawRecord`：保留数据集原始字段，不强制统一。
- `CanonicalCycle`：统一到 cell-cycle 层级，字段允许 optional。
- `TrainingWindow`：模型真正读取的固定张量格式，由配置决定使用哪些通道。

### 7.3 Severson parser 第一版任务

第一版 Severson parser 只需要完成：

1. 读取 `.pkl` 或 `.mat`。
2. 展开为 cell-level 和 cycle-level 表。
3. 保存 `cell_id`、`batch_id`、`cycle_index`。
4. 提取 `summary.QD` 作为容量/SoH 主标签。
5. 提取 cycle 内 `I`、`V`、`T`、`Qc`、`Qd`、`Qdlin`、`Tdlin`、`dQdV`、`t`。
6. 保存原论文 train/test/secondary_test split。
7. 保存一个小样本供本地调试。

## 8. 服务器下载后的最小回传内容

下载完整数据后，建议先从服务器回传：

```text
README 或 metadata 文件
三个 batch 文件名和大小
1 个 batch 的前 1-3 个 cell
每个 cell 的 summary
每个 cell 的前 5-10 个 cycles
```

如果 `.pkl` 已经存在，优先回传 `.pkl` 小样本；如果只有 `.mat`，可以在服务器上用作者 notebook 的逻辑抽样后再回传。
