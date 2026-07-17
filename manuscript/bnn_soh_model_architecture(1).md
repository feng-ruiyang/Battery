# BNN–Transformer 电池 SoH 概率轨迹模型

> 更新版：以 VBLL + 显式退化动力学为主方案，同时保留 Monte Carlo BNN、SoH 自回归和 latent dynamics 作为备选路线、消融对象与强基线。

## 1. 研究目标

BNN（Bayesian Neural Network，贝叶斯神经网络）用于描述模型参数不确定性；SoH（State of Health，健康状态）为主要预测对象。

给定截至循环 $t$ 的电压、电流、温度及工况数据，输出当前和未来 SoH 的联合预测分布：

$$
p(\mathbf y_{t:t+H}\mid \mathbf X_{1:t},\mathbf u_{1:t},\mathcal D),
\qquad \mathbf y=\mathrm{SoH}.
$$

最终输出不仅包括点预测，还包括：

- SoH 后验预测轨迹，即“概率云”；
- RUL（Remaining Useful Life，剩余使用寿命）分布；
- EOL（End of Life，寿命终点）首次越过阈值的循环分布；
- epistemic uncertainty（模型认知不确定性）与 aleatoric uncertainty（数据随机不确定性）的可解释分解。

主模型不使用 Kalman 滤波。Kalman 类方法不作为本项目的理论基础，但可在后续工作中作为外部基线。

## 2. 路线优先级

| 路线 | 作用 | 当前优先级 |
|---|---|---:|
| VBLL + 显式 SoH 退化动力学 | 单次前向、解析输出联合轨迹分布 | 主方案 |
| Monte Carlo 变分 BNN | 更灵活地近似权重后验，检验 VBLL 的表达上限 | 保留备选 |
| Bayesian SoH 自回归 Decoder | teacher forcing + rollout，研究训练—推理偏差 | 保留消融 |
| Bayesian latent dynamics | 预测 $z_{t+1}$ 并用预训练 Decoder 解码 | 高成本扩展 |
| Last-layer Laplace | 低成本近似贝叶斯推理 | 备用基线 |

主方案优先解决计算效率、概率校准和与既有 battery world model 的方法区分。其他路线不删除：如果主方案出现欠拟合、分布表达不足或动力学约束过强，可按上述顺序启用。

## 3. 共享输入与表示学习

### 3.1 输入

- 单循环信号：$X_t\in\mathbb R^{L\times d_x}$，包含电压、电流、温度等序列；
- 辅助变量：$u_t$，包含循环次数、环境温度、倍率和可用工况；
- 历史窗口：最近 $W$ 个循环；
- 预测范围：未来 $H$ 个循环。

### 3.2 阶段 A：Encoder–Decoder 生成式预训练

Transformer Encoder 将单循环信号编码为：

$$
z_t=E_\phi(X_t),\qquad z_t\in\mathbb R^{d_z}.
$$

Transformer Decoder 自回归重构该循环信号：

$$
p_\psi(X_t\mid z_t)
=\prod_{\ell=1}^{L}p_\psi(x_{t,\ell}\mid x_{t,<\ell},z_t).
$$

预训练目标为：

$$
\mathcal L_{\mathrm{pre}}
=-\log p_\psi(X_t\mid E_\phi(X_t)).
$$

预训练完成后：

1. 先冻结 $E_\phi$，完成下游预测头 warm-up；
2. 主模型稳定后，可仅解冻 Encoder 顶部一至两层，以较小学习率联合微调；
3. 联合微调时保留小权重重构损失，避免表示完全偏离原信号空间；
4. 预训练 Decoder 默认不参与主方案推理，但在 latent dynamics 支线中保留使用。

### 3.3 历史特征聚合

使用确定性的小型 Transformer Encoder 或时间序列聚合器处理最近 $W$ 个循环：

$$
h_t=A_\eta(z_{t-W+1:t},u_{t-W+1:t}),
\qquad h_t\in\mathbb R^{d_h}.
$$

这里 $h_t$ 是 Bayesian 预测层的输入。$A_\eta$ 默认采用普通点估计权重，不纳入主方案的 Bayesian 参数范围。

## 4. 主方案：VBLL + 显式退化动力学

VBLL（Variational Bayesian Last Layer，变分贝叶斯最后一层）只将最终线性层设为 Bayesian 层，其余网络保持确定性。

### 4.1 最后一层输出什么

先用一个普通 FFN（Feed-Forward Network，前馈网络）变换历史表示：

$$
\varphi_t=\operatorname{FFN}(h_t).
$$

Bayesian Linear 输出当前 SoH 与 $K$ 个低维动力学系数：

$$
\theta=[s_t,c_1,\ldots,c_K]^\top.
$$

普通线性层为：

$$
\theta=W\varphi_t+b.
$$

主方案把最后一层权重写成 mean-field 变分后验：

$$
q(W)=\prod_{i,j}\mathcal N(W_{ij};M_{ij},\sigma_{ij}^2),
\qquad
\sigma_{ij}=\operatorname{softplus}(\rho_{ij}).
$$

因此实际训练的 Bayesian 参数是：

- 权重均值矩阵 $M$；
- 权重尺度参数 $\rho$；
- bias 的均值与尺度参数；
- 可选的 aleatoric noise 参数。

第一版不训练巨大的完整权重协方差矩阵，而使用对角/mean-field 后验，控制计算量。

### 4.2 单次前向的解析分布

在独立权重近似下：

$$
\mu_\theta=M\varphi_t+b_\mu,
$$

$$
v_\theta
=\sigma_W^2(\varphi_t\odot\varphi_t)+\sigma_b^2.
$$

于是：

$$
q(\theta\mid h_t)
=\mathcal N\!\left(
\mu_\theta,
\operatorname{diag}(v_\theta)
\right).
$$

这一步不需要重复采样或多次运行 Transformer。

### 4.3 显式 SoH 退化动力学

用固定或低参数量的 basis matrix 将 $K$ 个系数展开成未来 $H$ 步的退化速率：

$$
\mathbf r=B\mathbf c,
\qquad
B\in\mathbb R^{H\times K},
\qquad
\mathbf c=[c_1,\ldots,c_K]^\top.
$$

$B$ 可使用 B-spline、分段线性 basis 或其他平滑基函数，使较少的系数能够表示：

- 早期缓慢退化；
- degradation knee；
- 后期加速下降；
- 不同工况下的退化速率变化。

未来 SoH 通过累计退化速率得到：

$$
y_{t+h}=s_t-\sum_{j=1}^{h}r_j+\epsilon_h.
$$

写成矩阵形式：

$$
\mathbf y_{t:t+H}=G\theta+\boldsymbol\epsilon.
$$

因此完整轨迹分布可解析传播为：

$$
\mu_{\mathbf y}=G\mu_\theta,
$$

$$
\Sigma_{\mathbf y}
=G\operatorname{diag}(v_\theta)G^\top
+\Sigma_{\mathrm{ale}}.
$$

其中：

- $G\operatorname{diag}(v_\theta)G^\top$ 表示最后一层参数不确定性向整条轨迹的传播；
- $\Sigma_{\mathrm{ale}}$ 表示测量噪声、短期容量恢复与不可预测退化扰动；
- 即使 $v_\theta$ 为对角形式，经过同一个 $G$ 后，不同未来循环仍会自然产生相关性。

### 4.4 动力学软约束

不强制观测 SoH 每一步严格单调，而只约束长期退化趋势。可选正则项包括：

$$
\mathcal L_{\mathrm{neg}}
=\frac1H\sum_{h=1}^{H}\max(0,-r_h)^2,
$$

用于惩罚明显为负的退化速率；以及：

$$
\mathcal L_{\mathrm{smooth}}
=\frac1{H-2}\sum_{h=2}^{H-1}
(r_{h+1}-2r_h+r_{h-1})^2,
$$

用于避免退化率无意义地快速振荡。短期容量恢复由 $\boldsymbol\epsilon$ 或单独 residual 分支描述。

## 5. 主方案训练目标

### 5.1 不需要动力学系数标签

训练数据只需要真实 SoH 轨迹 $\mathbf y_{t:t+H}$。动力学系数 $\theta$ 是网络内部变量，通过 $G$ 映射为轨迹并接受端到端监督：

$$
h_t
\rightarrow
(\mu_\theta,v_\theta)
\rightarrow
(\mu_{\mathbf y},\Sigma_{\mathbf y})
\rightarrow
\mathbf y_{t:t+H}.
$$

### 5.2 联合损失

主损失使用多元高斯 NLL（Negative Log-Likelihood，负对数似然）：

$$
\mathcal L_{\mathrm{NLL}}
=\frac12(\mathbf y-\mu_{\mathbf y})^\top
\Sigma_{\mathbf y}^{-1}
(\mathbf y-\mu_{\mathbf y})
+\frac12\log|\Sigma_{\mathbf y}|.
$$

总损失为：

$$
\mathcal L_{\mathrm{main}}
=\mathcal L_{\mathrm{NLL}}
+\beta\,KL(q(W)\Vert p(W))
+\lambda_{\mathrm{neg}}\mathcal L_{\mathrm{neg}}
+\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}.
$$

其中 $\mathcal L_{\mathrm{rec}}$ 仅在部分解冻 Encoder 时启用。$\Sigma_{\mathrm{ale}}$ 应加入最小对角 jitter，保证 $\Sigma_{\mathbf y}$ 数值上正定。

### 5.3 推荐训练顺序

1. 训练 Encoder–Decoder 重构模型；
2. 冻结 Encoder，训练确定性历史聚合器与确定性动力学头，得到稳定基线；
3. 将最后一层替换为 VBLL，训练 $M$、$\rho$ 与 noise 参数；
4. 加入动力学软约束并调节正则权重；
5. 可选：解冻 Encoder 顶部一至两层，以较小学习率联合微调；
6. 完成概率校准后，再决定是否启用 Monte Carlo 或 latent dynamics 支线。

## 6. 保留路线 A：Monte Carlo 变分 BNN

MC（Monte Carlo，蒙特卡洛）仍然保留，主要用于以下情形：

- VBLL 的 mean-field 后验表达不足；
- 需要让 FFN 或动力学模块的更多层具有 Bayesian 权重；
- 需要检验仅最后一层 Bayesian 化是否低估 epistemic uncertainty；
- 作为 VBLL 的重要概率基线。

设 Bayesian 权重为：

$$
W\sim q_\theta(W).
$$

训练时用 $S$ 次权重采样近似期望：

$$
\mathcal L_{\mathrm{MC-BNN}}
=\frac1S\sum_{s=1}^{S}
\left[-\log p(\mathbf y\mid\mathbf z,\mathbf u,W^{(s)})\right]
+\beta KL(q_\theta(W)\Vert p(W)).
$$

可测试三种范围：

1. 仅最后一层 Bayesian；
2. 最后一个 FFN block + 输出层 Bayesian；
3. 完整小型预测头 Bayesian。

训练期可先使用较小的 $S$，推理期再增加样本数。MC Dropout、Bayes by Backprop 和局部重参数化可分别作为实现与消融。所有样本应尽量并行化，避免逐样本串行前向。

## 7. 保留路线 B：Bayesian SoH 自回归 Decoder

该路线保留原始计划，用于研究直接自回归建模与 exposure bias：

$$
(\mu_{t+1},\sigma^2_{\mathrm{ale},t+1})
=f_W(z_{t-W+1:t},y_{t-W+1:t},u_{t-W+1:t+1}).
$$

推荐使用小型因果 Transformer Decoder：

- 训练初期使用 teacher forcing；
- 后期增加多步 rollout 微调；
- 训练损失包含单步 NLL 与多步 rollout loss；
- 推理时每条轨迹采样一次 $W^{(m)}$，并在整条预测区间保持该权重不变；
- 每一步可独立采样 aleatoric noise。

损失为：

$$
\mathcal L_{\mathrm{AR-BNN}}
=\mathbb E_{q_\theta(W)}[-\log p(\mathbf y\mid\mathbf z,\mathbf u,W)]
+\beta KL(q_\theta(W)\Vert p(W))
+\lambda\mathcal L_{\mathrm{rollout}}.
$$

该路线的主要风险是训练—推理偏差与长期误差累积，因此不作为当前第一实现，但必须保留为结构消融。

## 8. 保留路线 C：Bayesian latent dynamics

该路线预测下一循环 latent state：

$$
\hat z_{t+1}
=F_W(z_{t-W+1:t},u_{t-W+1:t+1}).
$$

冻结 Encoder 提供伪监督标签：

$$
z_{t+1}=E_\phi(X_{t+1}).
$$

为避免 latent state 只对 SoH 有效而缺乏状态一致性，至少加入：

$$
\mathcal L_z
=\|\hat z_{t+1}-E_\phi(X_{t+1})\|^2,
$$

以及使用预训练 Decoder 的观测重构约束：

$$
\mathcal L_X
=\|D_\psi(\hat z_{t+1})-X_{t+1}\|^2.
$$

再通过 SoH head 得到：

$$
\hat y_{t+1}=g(\hat z_{t+1}).
$$

该路线与已有 battery world model 的“latent transition + iterative rollout”主干高度接近，因此定位为：

- 对主方案的高成本扩展；
- 用于验证显式 SoH 动力学与 latent dynamics 的差异；
- 若采用，贡献必须集中在 Bayesian transition、latent consistency、观测重构和概率校准，而不能宣称首次使用 battery latent world model。

## 9. 概率云生成

### 9.1 主方案

均值与协方差可直接解析获得：

$$
\mathbf y_{t:t+H}
\sim
\mathcal N(\mu_{\mathbf y},\Sigma_{\mathbf y}).
$$

如果只报告均值和预测区间，不需要 MC。若需要可视化概率云或计算 EOL 首次越界分布，则从低维 $\theta$ 或最终多元高斯中向量化采样：

$$
\theta^{(m)}\sim
\mathcal N(\mu_\theta,\Sigma_\theta),
\qquad
\mathbf y^{(m)}=G\theta^{(m)}+\epsilon^{(m)}.
$$

这类采样只涉及高斯采样和矩阵乘法，不需要重复运行 Transformer。

### 9.2 Monte Carlo BNN 与自回归路线

对每条轨迹 $m=1,\ldots,M$：

1. 采样一次权重 $W^{(m)}\sim q_\theta(W)$；
2. 在整个预测区间内保持该权重不变；
3. 生成完整轨迹，而不是每一步重新采样模型身份；
4. 单独采样 aleatoric noise；
5. 汇总得到：

$$
\mathcal Y
=\{\mathbf y_{t:t+H}^{(1)},\ldots,
\mathbf y_{t:t+H}^{(M)}\}.
$$

## 10. 最终输出

- 当前及未来 SoH 的均值、中位数和预测区间；
- $M$ 条联合 SoH 概率轨迹；
- $P(\mathrm{SoH}_{t+h}<80\%)$ 随 horizon 的变化；
- EOL 首次越过阈值的循环分布；
- RUL 后验分布与可信区间；
- epistemic / aleatoric uncertainty 分解；
- 推理耗时与概率质量之间的权衡。

## 11. 实验计划

### 11.1 主数据集与公平比较

主实验优先使用 Severson LFP 数据集，以便与最接近的 battery world model 直接比较：

- 使用 cell-level 数据划分，严禁同一电池的循环跨训练集与测试集；
- 对齐该工作的 $W=30$ 输入窗口与 $H=80$ 预测长度；
- 输入优先保持为电压、电流、温度及同等可用工况；
- 若无法获得对方的精确 cell ID 划分，应固定公开划分并报告多个随机种子；
- 将原文报告结果与本项目复现结果分开列出。

### 11.2 必要基线

- Lim and See 的 deterministic World Model / Physics-Informed World Model；
- CNN–PatchTST 直接多步回归；
- 相同 backbone 的确定性动力学头；
- VBLL，无显式动力学；
- VBLL + 显式退化动力学；
- MC Dropout；
- Deep Ensemble；
- Bayes by Backprop BNN；
- Last-layer Laplace；
- 可选：SoH 自回归 Decoder 与 Bayesian latent dynamics。

### 11.3 消融实验

- 去除生成式预训练；
- Encoder 完全冻结 vs. 部分解冻；
- 去除 Bayesian 层；
- VBLL vs. MC-BNN；
- 去除显式动力学矩阵 $G$；
- 不同 $K$ 与 basis 形式；
- 去除 $\mathcal L_{\mathrm{neg}}$；
- 去除 $\mathcal L_{\mathrm{smooth}}$；
- 对角 aleatoric covariance vs. 低秩 covariance；
- 解析直接轨迹 vs. SoH 自回归 rollout vs. latent rollout；
- 去除循环次数或工况变量。

### 11.4 评价指标

点预测：

- MAE（Mean Absolute Error，平均绝对误差）；
- RMSE（Root Mean Squared Error，均方根误差）；
- 不同 horizon 的误差曲线。

概率质量：

- NLL；
- CRPS（Continuous Ranked Probability Score，连续排名概率分数）；
- 50% / 80% / 90% / 95% 区间覆盖率；
- 在相同覆盖率下的区间宽度；
- reliability diagram 与 calibration error。

泛化与可靠性：

- leave-one-cell-out；
- 跨充电工况；
- OOD（Out-of-Distribution，分布外）测试；
- EOL / RUL 误差与区间覆盖率；
- 至少 3 个随机种子，并以电池为单位报告置信区间。

计算效率：

- 训练时间；
- 单次推理时间；
- 生成 $M$ 条轨迹的时间；
- 参数量和峰值显存；
- VBLL、MC-BNN 与 Deep Ensemble 的成本对比。

## 12. 分阶段实验决策

### P0：确定性可行性验证

- 完成数据处理、Encoder 和确定性轨迹预测头；
- 确认没有数据泄漏；
- 对齐 World Model 论文的基本设置和指标。

### P1：主方案最小实现

- 将最后一层替换为 VBLL；
- 使用较小的 $K$ 和对角后验；
- 验证 NLL、覆盖率和数值稳定性；
- 与相同 backbone 的确定性模型比较。

### P2：加入显式动力学

- 加入 $B$、$G$ 与动力学软约束；
- 比较 VBLL 直接输出与动力学参数化输出；
- 检查 degradation knee 与长期预测。

### P3：启用 Monte Carlo BNN

仅在以下情况启用：

- VBLL 明显欠拟合；
- OOD 下不确定性严重偏小；
- 最后一层不确定性不足以解释预测误差；
- 计算预算允许更广的 Bayesian 参数范围。

### P4：启用 latent dynamics

仅当主方案已得到稳定结果，且 latent consistency 与重构损失能够带来明确增益时，再将其升级为主要实验支线。

## 13. 论文定位

主方案定位为：

**生成式预训练 Transformer 表示 + sampling-free Bayesian last layer + 显式健康空间退化动力学 + 解析轨迹级不确定性传播与校准。**

与已有 battery world model 的核心区别：

- 不传播不可解释的 latent state，而是在可解释的 SoH / 退化率空间建模；
- 不依赖 80 步确定性 latent rollout，而是单次前向得到联合轨迹分布；
- 显式区分预测均值、权重不确定性和数据噪声；
- 使用概率评分、区间覆盖率和 EOL / RUL 分布评价模型可靠性。

边界声明：若只将输出头 Bayesian 化，所得 epistemic uncertainty 主要描述给定确定性表示后的预测头不确定性，不包含冻结 Encoder 的完整表示不确定性。论文中应明确这一范围，不能将其表述为整个模型的完整 Bayesian posterior。

## 14. 关键参考方向

- Variational Bayesian Last Layers：sampling-free、single-pass 的最后一层变分贝叶斯方法；
- Weight Uncertainty in Neural Networks：Bayes by Backprop 与 MC 变分训练；
- Laplace Redux：最后一层 Laplace 近似；
- World Model for Battery Degradation Prediction Under Non-Stationary Aging：最接近的 deterministic latent rollout 基线。
