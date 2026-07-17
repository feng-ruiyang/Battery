建议你按“从数据到模型，再到训练”的顺序读，不要从 `trainer.py` 开始硬啃。现在最重要的是先理解数据怎么变成模型能吃的格式。

**1. 先读数据结构**

先看：

[code/src/battery_soh/data/schema.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/data/schema.py)

重点理解这三个类：

```python
CanonicalCell
CanonicalCycle
TrainingWindowSpec
```

你要先搞清楚：

```text
CanonicalCell  = 一个电池一行
CanonicalCycle = 一个电池的一个循环一行
TrainingWindow = 未来真正喂给模型的一段历史窗口
```

**2. 再读 Severson parser**

看：

[code/src/battery_soh/data/severson.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/data/severson.py)

重点读这几个函数：

```python
SeversonParser.scan()
SeversonParser.parse()
_parse_beep_structure()
_group_interpolated_cycles()
```

这部分回答的问题是：

```text
FastCharge_*.json
  -> cells.jsonl
  -> cycles.jsonl
  -> arrays/cell_id/cycle_xxxxx.npz
```

尤其看 `_group_interpolated_cycles()`，它对应你服务器上看到的扁平表：

```text
cycles_interpolated.cycle_index = [0,0,0,1,1,1,...]
```

代码会按 `cycle_index` 分组，变成每个 cycle 一个 `.npz`。

**3. 看预处理入口**

看：

[code/scripts/preprocess.py](/Users/fry/PaperProject/Battery/code/scripts/preprocess.py)

这文件很短，作用是：

```text
读取参数
创建 SeversonParser
scan 原始目录
parse 到 output_dir
```

你要理解以后服务器上跑的命令最终就是调用这里。

**4. 看数据读取和窗口**

看：

[code/src/battery_soh/data/datasets.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/data/datasets.py)

重点看：

```python
CanonicalCycleStore
SOHTrajectoryDataset
```

这部分还没完全接训练，但它的目标是：

```text
cells.jsonl + cycles.jsonl + arrays/*.npz
  -> 训练样本 TrainingWindow
```

**5. 再看模型总入口**

看：

[code/src/battery_soh/models/soh_predictor.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/models/soh_predictor.py)

先不要看每个模块细节。先理解总流程：

```python
z_window
  -> HistoryAggregator
  -> VBLLLinear / DeterministicHead
  -> DegradationDynamics
  -> y_mean, y_cov
```

这就是主模型主干。

**6. 看历史聚合器**

看：

[code/src/battery_soh/models/history_aggregator.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/models/history_aggregator.py)

它做的是：

```text
过去 W 个 cycle 的 latent z
  -> Transformer Encoder
  -> 当前历史状态 h_t
```

你重点看 shape：

```python
z_window: [B, W, latent_dim]
h_t:      [B, hidden_dim]
```

**7. 看 VBLL**

看：

[code/src/battery_soh/models/vbll.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/models/vbll.py)

这部分稍微难一点。你先只理解输出：

```python
theta_mean
theta_var
kl
```

意思是：

```text
theta_mean = 当前 SoH + 退化系数的均值
theta_var  = 这些参数的不确定性
kl         = 贝叶斯正则项
```

**8. 看退化动力学**

看：

[code/src/battery_soh/models/degradation_dynamics.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/models/degradation_dynamics.py)

重点看：

```python
make_piecewise_linear_basis()
make_trajectory_matrix()
DegradationDynamics.forward()
```

它做的是：

```text
theta = [s_t, c1, c2, ..., cK]
  -> 未来退化速率
  -> 累计退化
  -> y_mean, y_cov
```

这部分对应 proposal 里的显式退化动力学。

**9. 看 loss**

看：

[code/src/battery_soh/training/losses.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/training/losses.py)

重点看：

```python
gaussian_trajectory_nll()
total_loss()
```

它做的是：

```text
预测分布 y_mean/y_cov
和真实 soh_future 比较
得到 NLL + KL + 正则
```

**10. 最后看 trainer**

最后再看：

[code/src/battery_soh/training/trainer.py](/Users/fry/PaperProject/Battery/code/src/battery_soh/training/trainer.py)

现在你打开的就是这个。它只是训练循环外壳：

```python
for batch in dataloader:
    outputs = model(...)
    losses = total_loss(...)
    losses["loss"].backward()
    optimizer.step()
```

所以它应该最后看，不然会觉得一堆东西不知道从哪里来。

推荐阅读顺序总结：

```text
1. data/schema.py
2. data/severson.py
3. scripts/preprocess.py
4. data/datasets.py
5. models/soh_predictor.py
6. models/history_aggregator.py
7. models/vbll.py
8. models/degradation_dynamics.py
9. training/losses.py
10. training/trainer.py
```

读代码时每个文件只问三个问题：

```text
这个文件输入是什么？
输出是什么？
shape 怎么变？
```

先别追求一遍看懂所有细节。你先把数据流和 shape 串起来，后面看 Torch 细节会轻很多。