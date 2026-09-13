# AWM-AutoRoute 论文实验总规划

**状态：当前唯一有效的论文实验总表。**  
**方法：AWM-AutoRoute（唯一正式方法）。**  
**更新时间：2026-09-13。**

本文档同时定义实验范围、运行单元、状态和总进度。其他协议、历史运行
记录和旧计划只提供来源与审计证据，不再各自充当待办清单。

## 1. 计数和准入规则

一个完整训练单元定义为：

> 一个 benchmark × 一个方法/对照/消融 × 一个预声明 seed，完成全部六个
> 任务、规定更新预算和最终评估，并保存可审计 manifest 与原始逐任务指标。

状态：

- `COMPLETE`：完整训练已经跑完；结果仍需完成本地备份、registry 导入和
  协议/可比性审计，才可写入论文最终表；
- `RUN`：正在运行，或结果仍在远端等待导入与审计；
- `TODO`：尚未启动；
- `HIST`：历史/退休/失败/停止证据，不计入分母完成量。

只有完全相同的协议、benchmark、seed、预算和评估才可填一个格。同一格的
失败与重试全部保留，但最多计一次完成。Smoke、部分任务、旧路由、根据结果
挑选的 seed，以及 AWM/D oracle 训练都不能冒充 AWM-AutoRoute 正式结果。

## 2. 冻结的论文范围

### 2.1 主结果：30 个完整训练单元

目标：回答 AWM-AutoRoute 是否在 Atari 与 CoinRun 上同时改善学习、最终
保持和遗忘，并报告参数与存储代价。

| Benchmark | 方法 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | 小计 |
|---|---|---|---|---|---|---|---:|
| Atari | DreamerV3/FIFO | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 5 |
| Atari | ARROW-50 | COMPLETE | TODO | TODO | TODO | TODO | 5 |
| Atari | **AWM-AutoRoute** | RUN | RUN | RUN | RUN | RUN | 5 |
| CoinRun | DreamerV3/FIFO | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 5 |
| CoinRun | ARROW-50 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 5 |
| CoinRun | **AWM-AutoRoute** | TODO | TODO | TODO | TODO | TODO | 5 |

DreamerV3/FIFO、ARROW-50 和结构对照只是基线/对照。论文提出的方法只有
AWM-AutoRoute。2026-09-13 冻结主表的五个 Atari seed 为当时八个正在运行的
v4 pilot 中 seed 数值最小的五个：`175428134`（S24）、`238229439`（S25）、
`671523726`（S27）、`1171886581`（S22）、`1210121915`（S28）。这是一条与
性能无关的固定选择规则；五格不得按结果替换。其余三个正在运行的 S16
(`1527319854`)、S29 (`2188084315`) 和 S30 (`3571757014`) 作为额外稳健性证据，
所有成功、失败和停止结果仍须在附录披露，但不进入 57 的分母。主表使用非
配对多 seed 汇总，不进行伪配对检验。

### 2.2 容量组织对照：15 个完整训练单元

目标：排除“只是参数更多、只是 private actor、只是完整隔离”的替代解释。
范围固定为 Atari 三 seeds；这些对照是 task-aware 结构诊断，不与
task-ID-free 的 AWM-AutoRoute 冒充完全同协议排名。

| 对照 | S0 (`123456789`) | S1 (`1337`) | S2 (`31337`) | 小计 |
|---|---|---|---|---:|
| Shared WM + private AC | RUN | RUN | TODO | 3 |
| Wider shared + private AC | RUN | RUN | TODO | 3 |
| FullBank + private AC | COMPLETE | RUN | RUN | 3 |
| Frozen core + residuals | COMPLETE | RUN | TODO | 3 |
| Independent residuals | COMPLETE | RUN | TODO | 3 |

这里采用已批准的 `CapacityOrganization-v1-Atari` 协议。旧的 Task-0、三任务
FullBank 以及其他退休结构不填这些格。

### 2.3 机制消融：12 个完整训练单元

目标：分别验证复用、功能保护、冲突处理和自适应压缩。范围固定为 Atari
三 seeds。每一项必须先冻结单变量协议和计算预算，再允许启动。

| AWM-AutoRoute 变体 | S0 | S1 | S2 | 小计 |
|---|---|---|---|---:|
| NoReuse：关闭历史机制复用 | TODO | TODO | TODO | 3 |
| NoFunctionalProtection：关闭旧功能保持目标 | TODO | TODO | TODO | 3 |
| NoConflictProjection：关闭冲突梯度投影 | TODO | TODO | TODO | 3 |
| NoRCC：关闭 return-gated structural compaction | TODO | TODO | TODO | 3 |

消融不得同时改变交互步数、Replay 容量、在线更新数、评估机会或任务标签。
若关闭 RCC 会减少边界更新，协议必须预先规定计算匹配方式或明确标成非计算
匹配，不能跑完后再选择解释。

## 3. 不增加完整训练单元的评估与图表

以下工作复用已完成 checkpoint；分别跟踪完成状态，但不进入 57 个训练分母：

| 产物 | 数据/评估 | 状态 |
|---|---|---|
| 主结果表 | Acquisition、Final、Forgetting、raw per-task returns、retained params/bytes | TODO |
| 持续学习曲线 | 四阶段/六任务曲线与 task × stage retention matrix | TODO |
| 容量 Pareto 图 | Final performance 对 retained parameters/bytes | TODO |
| Oracle routing 上界 | 同一 AWM-AutoRoute checkpoint 的 oracle-route 与 auto-route 评估；不是第二个方法 | TODO |
| 路由诊断 | route accuracy、confusion、return gap、reset latency | TODO |
| Historical reuse | gate/route 权重加功能干预后的性能变化 | TODO |
| Retained width | 每任务最终 Q/F/P 宽度及实际参数/字节 | TODO |
| Predictive retention | 代表任务上 `H=1,2,4,8,16` open-loop prediction error | TODO |
| 统计与报告 | 预声明聚合、置信区间/IQR、表图生成命令、原始指标追溯 | TODO |

Oracle routing 只是相同模型的诊断读出，名称写成
`AWM-AutoRoute (oracle-route evaluation)`，不得引入 `AWM-O` 或把 AWM
重新包装成第二个正式方法。

## 4. 当前总进度

冻结的完整训练总量：

\[
30\;\text{主结果}+15\;\text{结构对照}+12\;\text{机制消融}
=\boxed{57\;\text{个完整训练单元}}
\]

截至本文档建立时：

| 状态 | 数量 | 比例 |
|---|---:|---:|
| COMPLETE | 19 | 33.3% |
| RUN | 13 | 22.8% |
| TODO | 25 | 43.9% |
| **总计** | **57** | **100%** |

训练完成度为 `COMPLETE / 57 = 33.3%`，已有/在途覆盖率为
`(COMPLETE + RUN) / 57 = 56.1%`。`COMPLETE` 是确实跑完，不代表已经通过
最终论文可比性审计。七个刚完成的云端目录正在同步到本地；同步与校验结果
必须进入备份清单。

## 5. 执行顺序

1. 完成 19 个 `COMPLETE` 的本地备份、registry 导入和协议可比性审计；不匹配
   的格转 `TODO` 并写明原因，但不得声称从未运行。
2. 等待并导入 13 个 `RUN`；失败记录保留，未完成格仍是 `TODO`。
3. 补 Atari ARROW-50 S1–S4、CoinRun AWM-AutoRoute 五 seeds。
4. 补四个缺失的容量对照 S2。
5. 冻结四份单变量消融协议，通过测试与目标 GPU smoke 后运行 12 格。
6. 统一生成诊断、表格和图，不手工抄最终数字。

任何新增 benchmark、额外 seed、外部基线或细粒度超参数消融都先作为本计划的
显式修订，不得用空卡临时创造实验。

## 6. 云端结果备份规则

云端实例不是结果的唯一保存位置。每个运行遵守以下顺序：

1. 运行中定期拉取配置、manifest、日志和已有原始指标；
2. `run_status.json` 确认成功后，立即拉取完整结果目录，包括所有最终模型和
   task-boundary/analysis snapshots；
3. 本地逐文件核对大小和 SHA-256，写入备份 manifest；
4. 将小型结果摘要导入 `docs/experiments/records/` 并重建 registry；
5. 只有本地备份验证通过后，才允许释放或删除云端实例/目录。

大文件保存在被 Git 忽略的
`runs/cloud_result_backups/<date>/`；Git 只保存可审计的小型结果记录。当前七个
云端完整结果的日志、配置、指标和 manifest 已拉到本地，大型 `.pt` 文件正在
分块同步并作远端/本地 SHA-256 比对。运行中任务的四机元数据快照也已保存。

## 7. 明确不在当前分母中的内容

- AWM/D 作为独立论文方法；
- D-AutoRoute v1/v2、AWM-AutoRoute v3 和一帧路由结果；
- 已退休的 KAN、KARROW、MoE、旧 FullBank/LoRA/REC 等方法；
- smoke、单任务 gate、少于六任务的 partial run；
- CoinRun 结构对照和 CoinRun 机制消融；
- R2-Dreamer、Dream Rehearsal 或更多外部基线，除非在看到结果前正式修订本计划。

这些材料继续保留为 provenance，不参与当前 57 个训练单元的完成百分比。
