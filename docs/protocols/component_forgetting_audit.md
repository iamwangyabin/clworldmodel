# Continual World-Model Component Forgetting Audit

## 文档状态

- 状态：Pilot P1 训练与分析快照已完成；component-level 离线审计正在执行。
- 当前阶段：已冻结 DreamerV3/FIFO seed-0 的训练产物，正从 task-boundary
  snapshots 生成 held-out diagnostics 并做 checkpoint differencing。
- 第一目标：定位 DreamerV3/FIFO 在连续任务训练中哪里发生功能性遗忘。
- 第二目标：比较 ARROW-50 改变了哪些遗忘通道。
- 非目标：现在不提出 targeted replay，不根据单 seed 结果宣称新方法有效。

这项研究把一次正常的 continual training 作为实验对象。训练完成后，
使用任务边界 checkpoint 和永不进入 replay 的固定审计数据进行离线分析，
不为每个 component 单独重训一套模型。

## 当前实现状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| DreamerV3/FIFO matched-control launcher | 已完成 | `scripts/run_dv3_fifo_atari.py` |
| Task-boundary analysis snapshots | 已完成 | epochs 89, 179, 269, 359, 449, 539 |
| Final analysis snapshot | 已完成 | epoch 540，和第六任务边界分开保存 |
| Snapshot SHA-256 与原子写盘 | 已完成 | 每个 `.pt` 有相邻 checksum |
| Launch/config/log/status provenance | 已完成 | `launch.json`, config, `train.log`, `run_status.json` |
| Snapshot serialization runtime verification | 已完成 | 7 个实际写入的 `.pt` 与 sidecar 都已校验；加载 smoke 在审计启动时重复验证 |
| Held-out diagnostic-set collector | 已完成 | 6 个 task 各 256 个 natural chunks；只读 snapshots，不写 replay |
| Teacher-forced evaluator | 已完成 | deterministic posterior mode；不调用训练更新 |
| Open-loop evaluator | 已完成 | 固定真实 action，horizons 1, 2, 4, 8, 16 |
| Actor/critic/representation evaluator | 已完成 | 使用同一 observation history；actor KL/agreement、anchored critic MAE、linear CKA |
| Versioned report generator | 已完成 | 保存 raw per-chunk metrics、JSON summary 和 Markdown report |
| Paired conclusion-data reporter | 已完成 | `scripts/summarize_component_audit.py`；用 episode-cluster paired bootstrap 对比 `$C_i$` 与 C6 |

Analysis snapshot 保存 world model 与 actor-critic 权重，但不包含 replay、
optimizer、RNG 或 environment-schedule state，因此不可声称为等价的可恢复训练
checkpoint。它们足够支持本文定义的离线 checkpoint differencing。

### Pilot P1 已有产物

`dv3_fifo_original_s0_analysis` 已正常结束（epoch 540，return code 0），并在本地
`runs/dv3_fifo_original_s0_analysis/` 保存了完整的 log、TensorBoard、7 份 snapshot
及全目录 checksum manifest。它提供单个 DreamerV3/FIFO seed 的 pilot 数据：可用于
验证诊断指标、发现候选遗忘通道，但不能作为跨 seed 的论文结论。

该训练启动时的 worktree 不是 clean，因此在当前项目治理规则下必须标记为 `pilot`，
不得被重新描述为 official baseline。此限制不影响其作为 evaluator 工程验证和假设
生成数据的价值。

## 核心研究问题

### RQ1：哪里发生遗忘？

当旧任务 return 下降时，退化首先出现在以下哪个位置：

1. observation encoder / posterior representation；
2. one-step latent prior；
3. long-horizon latent dynamics；
4. reconstruction decoder；
5. reward predictor；
6. continuation predictor；
7. actor；
8. critic；
9. representation 与 actor/heads 之间的读取接口。

### RQ2：遗忘是否依赖 rollout horizon？

短期预测可能保持稳定，而长时域 rollout 已经失真。需要回答：

\[
F_{\mathrm{dyn}}(H=1)
\quad\text{与}\quad
F_{\mathrm{dyn}}(H=16)
\]

是否表现出不同的遗忘速度。

### RQ3：ARROW 改变了遗忘量，还是改变了遗忘位置？

在完全匹配的训练和 replay-byte budget 下，比较：

\[
F_{\mathrm{DV3/FIFO}}(i,j,m,h)
\quad\text{和}\quad
F_{\mathrm{ARROW50}}(i,j,m,h).
\]

不预设 ARROW 一定保护 world model，也不预设 actor 一定是唯一失败点。

### RQ4：不同任务是否具有不同遗忘指纹？

如果某些任务主要丢失 long-horizon dynamics，另一些任务主要丢失 reward 或
actor competence，这种异质性才可能为后续 component-aware rehearsal 提供动机。

## 固定训练协议

### 方法

| 方法 | Replay | 总轨迹容量 | 总 observation 容量 | 角色 |
| --- | --- | ---: | ---: | --- |
| DreamerV3/FIFO | 1,024 FIFO | 1,024 | 524,288 | matched baseline/control |
| ARROW-50 | 512 FIFO + 512 LTDM | 1,024 | 524,288 | project primary method |

两种方法使用相同的 world model、actor、critic、任务顺序、环境交互预算、
world-model updates、actor-critic updates、reward scaling 和 evaluation schedule。
方法间只允许 replay retention/sampling 语义不同。

### Atari curriculum

| Task | Environment | Training epochs | Boundary snapshot |
| ---: | --- | ---: | --- |
| 1 | `ALE/MsPacman-v5` | 0-89 | `C1`, epoch 89 |
| 2 | `ALE/Boxing-v5` | 90-179 | `C2`, epoch 179 |
| 3 | `ALE/CrazyClimber-v5` | 180-269 | `C3`, epoch 269 |
| 4 | `ALE/Frostbite-v5` | 270-359 | `C4`, epoch 359 |
| 5 | `ALE/Seaquest-v5` | 360-449 | `C5`, epoch 449 |
| 6 | `ALE/Enduro-v5` | 450-539 | `C6`, epoch 539 |
| Final | schedule 已回到 Task 1 | 540 | `Cfinal`, epoch 540 |

上游配置使用 541 epochs，以便在 epoch 540 进行最后一次常规评估。但是 trainer
在该 epoch 已经重新收集并训练了一轮 Task 1，因此本文把 epoch 539 定义为真正
的第六任务完成 checkpoint，并单独报告 epoch 540，而不把两者混为一谈。

### Seed 与阶段

| 阶段 | 方法 | Seeds | 用途 | 是否支持论文结论 |
| --- | --- | --- | --- | --- |
| 已有 run | ARROW-50 | `123456789` | return 与运行性能参考；无 checkpoint | 否 |
| Pilot P1 | DreamerV3/FIFO | `123456789` | 验证完整审计管线、发现指标问题 | 否 |
| Confirmatory C1 | DreamerV3/FIFO | 5 个冻结 seeds | baseline component-forgetting estimate | 是 |
| Confirmatory C2 | ARROW-50 | 相同 5 seeds | matched method comparison | 是 |

不得因为 pilot seed 没有出现想要的遗忘现象而换 seed。负结果仍属于实验记录。

## 固定 diagnostic set

对每个任务 \(T_i\)，从其任务完成 checkpoint \(C_i\) 收集固定审计集：

\[
D_i=\{x_t,a_t,r_t,c_t,\mathrm{reset}_t\}.
\]

### Primary natural audit set

- 每个任务 256 个固定 chunks；
- 每个 chunk 64 agent decisions；
- 前 16 steps 只用于 RSSM burn-in；
- 后续位置用于 teacher-forced 与 open-loop 测量；
- collection 使用 \(C_i\) 的 stochastic policy，但环境 seed、actor sampling seed、
  episode order 和 chunk-selection seed 全部固定并记录；
- chunks 不跨 episode；终止后的 padding 必须用 valid mask 排除；
- observation 以 `uint8` 保存，evaluation 时应用冻结的原始 preprocessing；
- 同时保存 raw reward 与训练使用的 scaled reward；
- 保存 action、terminated、truncated、continue、reset、episode ID 和时间索引；
- 数据从未进入 replay，不影响模型更新、优先级、early stopping 或方法选择。

### Event audit subset

Reward 和 termination 事件在自然分布中可能稀少。可以从同一批 held-out episodes
确定性派生 event-centered subset，用于 reward/continue discrimination 与 calibration。
该 subset 只补充相应 head 的诊断，不代替自然分布上的 headline metric，也不得
把 event-balanced 数值解释为真实环境频率。

每个 dataset 保存独立 manifest，包括 checkpoint checksum、环境和 actor seeds、
任务顺序、collection policy、shape/dtype、preprocessing、reward scale 和文件 hash。

为避免一个极长的 Atari episode 阻塞自然分布审计，collector 对单个 collection
segment 使用显式的 decision cap。达到 cap 而未发生真实 `terminated`/`truncated`
时，该 segment 仍只能贡献全 `continue=1` 的 natural chunks；绝不会被伪造成
terminal event。每个 task manifest 记录已完成环境 episode 数、capped nonterminal
segment 数和 cap 值。Pilot P1 首先完成 natural headline tensor；event subset 仅在
能够收集到足量真实终止样本时作为补充运行。

## Checkpoint audit tensor

对所有 \(j\geq i\)，在相同 \(D_i\) 上评估 \(C_j\)：

\[
A_{i,j,m,h}=S_m(C_j,D_i,h),
\]

其中 \(m\) 是 component/metric，\(h\) 仅用于 rollout metric。最终数据结构为：

\[
\boxed{
\text{method}\times\text{seed}\times\text{old task}\times
\text{checkpoint}\times\text{component}\times\text{horizon}
}
\]

所有 posterior 和 prior 的离线评估使用 deterministic categorical mode。随机版本
只能作为单独的 robustness analysis，不能混入 headline table。

## Component metrics

| 位置 | Evaluation mode | Primary metrics | 可以支持的解释 | 不能单独证明的内容 |
| --- | --- | --- | --- | --- |
| Encoder/posterior | Teacher-forced | linear CKA, Procrustes residual | latent geometry 是否漂移 | representation drift 等于功能遗忘 |
| Prior/one-step dynamics | Posterior burn-in + one prior step | unclipped posterior-prior KL, prior-decoded prediction error | one-step predictability 是否退化 | 仅凭 KL 判定 transition 单独遗忘 |
| Long-horizon dynamics | Open-loop with fixed real actions | visual error, reward error, continue error by horizon | rollout 能力是否随 horizon 崩溃 | actor 导致的环境状态分布变化 |
| Decoder | Teacher-forced posterior | per-pixel MSE | 给定当前 latent 是否仍能识别旧 observation | dynamics 是否正确 |
| Reward head | Teacher-forced and open-loop | symlog MSE, raw/scaled MAE, event discrimination | reward knowledge 是否保留 | actor 是否还能到达奖励状态 |
| Continue head | Teacher-forced and open-loop | BCE, Brier score, terminal discrimination | termination knowledge 是否保留 | 整体 dynamics 是否正确 |
| Actor | Same raw history under each checkpoint | symmetric KL, top-1 agreement, old-action margin | 端到端 action mapping 是否漂移 | 纯 actor 参数是唯一原因 |
| Critic | Same raw history and fixed historical returns | value drift, anchored-return MAE | 旧轨迹价值信号是否仍可读 | 当前 policy 下严格的 value error |
| Latent-head interface | Frozen old readout on paired current latents | action/reward margin change | co-adaptation/interface 是否变化 | 未经 alignment 的跨 latent 因果归因 |

当前 vendored DreamerV3 中，`L_dyn` 与 `L_rep` 的 forward 数值来自同一个 KL；
`detach()` 只改变梯度流向。因此不能把两个相同 scalar 当成两个独立的离线
遗忘指标。需要用 functional metrics，或在后续 exploratory analysis 中分别检查
prior/transition 与 encoder/posterior 参数块的梯度。

## Forgetting definitions

对 loss/error 类指标，定义 boundary-relative forgetting：

\[
F_{i,j,m,h}=S_m(C_j,D_i,h)-S_m(C_i,D_i,h).
\]

对 CKA、agreement、discrimination 等越大越好的指标反转方向：

\[
F_{i,j,m,h}=S_m(C_i,D_i,h)-S_m(C_j,D_i,h).
\]

因此所有 headline \(F\) 都满足：正值表示退化，负值表示改善。必须同时保存并报告
原始绝对值，不允许只展示经过归一化的 forgetting score。

对环境 return 同时报告：

\[
F^{\mathrm{boundary}}_{i,j}=R_{i,i}-R_{i,j},
\]

以及标准 historical-maximum forgetting：

\[
F^{\max}_{i,j}=\max_{k\leq j}R_{i,k}-R_{i,j}.
\]

Raw per-task returns 是主数据。任何 normalized return 必须作为有固定常数和来源的
派生指标单独报告。

置信区间按 episode 聚类 bootstrap，不能把同一 episode 内的 timesteps 当成独立
样本。Confirmatory results 展示所有 per-seed points，不只展示跨 seed mean。

## 预注册解释矩阵

| 观测模式 | 暂时支持的解释 | 下一步最小验证 |
| --- | --- | --- |
| Teacher-forced 基本稳定，open-loop 随 horizon 明显恶化 | long-horizon dynamics candidate | 只针对 RSSM rollout 做验证性 intervention |
| Reconstruction/reward/continue 在 teacher-forcing 下共同恶化 | representation 或共享 model state candidate | 检查 latent drift、head margins 与参数块更新 |
| Reward head 单独恶化且 event discrimination 下降 | reward knowledge candidate | reward-head/shared-feature 最小 ablation |
| Continue head 单独恶化 | termination modeling candidate | termination-aware rollout validation |
| World-model 功能指标稳定，actor KL/return 同时恶化 | actor 或 latent-actor interface candidate | 区分 actor parameter drift 与 interface drift |
| CKA 大幅下降，但所有功能指标和 return 稳定 | benign representation drift | 不据此提出保护 representation 的方法 |
| 固定 \(D_i\) 上全部稳定，但 real return 下降 | state-distribution shift 或行为访问问题 | 增加隔离的 current-policy visit audit |
| Critic anchored error 恶化，actor 与 world model 稳定 | critic candidate | policy-dependence-aware value intervention |
| 不同任务由不同 component 主导 | heterogeneous forgetting | 才考虑 component-aware rehearsal |

这些模式是 inference rules，不是自动因果证明。真正的训练 ablation 只针对审计结果
排名最高的一到两个候选，避免重新训练所有 component 的组合爆炸。

## 可能的实验表格

以下表格全部是预先定义的模板。`TBD` 不是结果，不能用示例数字替换后当成证据。

### Table 1：End-to-end continual performance

| Method | Seed | Task | Return at acquisition | Best historical return | Final raw return | Boundary forgetting | Max forgetting | Final retention |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DV3/FIFO | 123456789 | MsPacman | TBD | TBD | TBD | TBD | TBD | TBD |
| DV3/FIFO | 123456789 | Boxing | TBD | TBD | TBD | TBD | TBD | TBD |
| ARROW-50 | 123456789 | MsPacman | TBD | TBD | TBD | TBD | TBD | TBD |
| ARROW-50 | 123456789 | Boxing | TBD | TBD | TBD | TBD | TBD | TBD |

正式报告包含全部六任务和全部 seeds。表中不得只选择遗忘最明显的任务。

### Table 2：Final component forgetting fingerprint

| Method | Task | Return forgetting | TF recon | 1-step prior | OL-H16 visual | OL-H16 reward | Continue | Actor symmetric KL | Critic calibration | Latent CKA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DV3/FIFO | MsPacman | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DV3/FIFO | Boxing | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ARROW-50 | MsPacman | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ARROW-50 | Boxing | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

每个单元格报告跨 seed mean、置信区间，并在图中保留 per-seed points。

### Table 3：Forgetting versus rollout horizon

| Method | Task | Metric | H=1 | H=2 | H=4 | H=8 | H=16 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DV3/FIFO | MsPacman | visual prediction error increase | TBD | TBD | TBD | TBD | TBD |
| DV3/FIFO | MsPacman | reward prediction error increase | TBD | TBD | TBD | TBD | TBD |
| ARROW-50 | MsPacman | visual prediction error increase | TBD | TBD | TBD | TBD | TBD |

### Table 4：One old task across all later checkpoints

| Audit set | Metric | C1 | C2 | C3 | C4 | C5 | C6 | Cfinal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D1` | Raw return | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `D1` | TF reconstruction | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `D1` | OL-H16 error | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `D1` | Actor symmetric KL from C1 | 0 | TBD | TBD | TBD | TBD | TBD | TBD |
| `D1` | Latent CKA with C1 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |

该表直接展示遗忘发生的时间顺序，而不只比较 acquisition 和 final 两个端点。

### Table 5：Matched ARROW effect by component

| Task | Component | DV3 forgetting | ARROW-50 forgetting | ARROW minus DV3 | 95% paired CI | Interpretation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| MsPacman | OL-H16 dynamics | TBD | TBD | TBD | TBD | TBD |
| MsPacman | Actor KL | TBD | TBD | TBD | TBD | TBD |
| Boxing | Reward head | TBD | TBD | TBD | TBD | TBD |

只有相同 seed、task、budget 和 audit set 才进入 paired comparison。

### Table 6：Minimal validation ablation

| Diagnostic finding | Validation arm | Changed component | Env steps | WM updates | AC updates | Replay bytes | Outcome |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| TBD | Control rerun | none | matched | matched | matched | matched | TBD |
| TBD | Candidate intervention | exactly one declared change | matched | matched | matched | matched | TBD |

该表只有在 diagnostic study 完成并预注册具体 candidate 后才能填充。

## 计划图形

1. Raw return matrix：task × checkpoint，分别展示 DV3/FIFO 和 ARROW-50。
2. Component forgetting heatmap：task × component，颜色表示 final forgetting。
3. Horizon curves：每个任务的 prediction degradation versus rollout horizon。
4. Selected-task trajectory：return、teacher-forced、open-loop、actor KL 随 checkpoint
   同图变化，但不通过视觉缩放夸大某个 component。
5. ARROW protection profile：ARROW minus DV3 的 paired component effects。

## 执行顺序

1. 对 analysis-snapshot 写入、checksum 和 CPU load 做微型 smoke test。
2. 冻结本协议与 seed-0 DreamerV3/FIFO launch manifest。
3. 运行 Pilot P1，确认六个 boundary snapshots 与 final snapshot 全部存在。
4. 从每个 \(C_i\) 生成 held-out \(D_i\)，冻结并 hash dataset manifest。
5. 实现纯离线 evaluator，首先验证同一 checkpoint 重复计算的确定性。
6. 生成 pilot 的 raw metrics、审计 tensor、表格和图，不据此提出正式方法结论。
7. 根据 pilot 修正 evaluator bug，但不得修改已经冻结的训练 protocol。
8. 冻结 confirmatory protocol，运行 DV3/FIFO 与 ARROW-50 的全部五个 seeds。
9. 只有在 component-level pattern 跨 seeds 和 tasks 成立后，才预注册最小
   intervention 或后续 targeted replay 方法。

## Definition of done

这项诊断研究只有在以下条件同时满足时才完成：

- 所有 run 的训练、交互、更新和 replay-byte budget 可核对；
- evaluation transitions 从未进入 replay；
- snapshots、datasets 和 raw metrics 有 checksum 与 manifest；
- 六任务 raw returns、final average performance 和 forgetting 全部保留；
- component metrics 在固定数据上可确定性复算；
- 结论遵守解释矩阵，不把相关性写成未经验证的因果结论；
- pilot、debug、failed 和 official runs 标签清楚；
- 没有根据有利结果选择或丢弃 seeds、tasks 或 checkpoints。
