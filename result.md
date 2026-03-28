# E1 / E2 Results

说明：
- `phase3_best` 是在 `phase2_best` 基础上继续进行 Phase 3 SFT 训练得到的权重。
- 本文件中的 E2 结果使用的是当前仓库 `models/` 下的完整 Phase 2 Fusion 模型，不是旧 Reference 的只加载 injection_modules 的评测子图。

## `E1` `p2_qwen3_10ep/phase2_best`

来源：
- [results/e1/e1_sanity_check_phase2_best.json](/home/undergraduate/zcy/Explicit-Lora/results/e1/e1_sanity_check_phase2_best.json)

| 指标 | 数值 |
|---|---:|
| acc_correct | 43.44% (553/1273) |
| acc_counterfactual | 29.07% (370/1273) |
| acc_no_knowledge | 31.97% (407/1273) |
| knowledge_sensitivity (KS) | +14.38% |

补充信息：

| 项目 | 值 |
|---|---|
| 权重 | `checkpoints/p2_qwen3_10ep/phase2_best` |
| 设备 | `cuda` |
| max_samples | -1 |
| total | 1273 |

## `E1` `p3_from_p2_qwen3_10ep/phase3_best`

来源：
- [results/e1/e1_sanity_check_p3_from_p2_qwen3_10ep_phase3_best.json](/home/undergraduate/zcy/Explicit-Lora/results/e1/e1_sanity_check_p3_from_p2_qwen3_10ep_phase3_best.json)

| 指标 | 数值 |
|---|---:|
| acc_correct | 70.07% (892/1273) |
| acc_counterfactual | 13.67% (174/1273) |
| acc_no_knowledge | 31.58% (402/1273) |
| knowledge_sensitivity (KS) | +56.40% |

补充信息：

| 项目 | 值 |
|---|---|
| 权重 | `checkpoints/p3_from_p2_qwen3_10ep/phase3_best` |
| 设备 | `cuda` |
| max_samples | -1 |
| total | 1273 |

## `phase2_best`（10轮，有 norm 训练版本）

来源：
- [results/e2/e2_cross_domain_phase2_best.json](/home/undergraduate/zcy/Explicit-Lora/results/e2/e2_cross_domain_phase2_best.json)

| 数据集 | Baseline | Fusion+知识 | Fusion+空知识 | Δacc | Δacc_empty |
|---|---:|---:|---:|---:|---:|
| MedQA（1273） | 32.91% (419/1273) | 42.58% (542/1273) | 33.46% (426/1273) | +9.66% | +0.55% |
| ARC-Challenge（1165） | 51.85% (604/1165) | 63.18% (736/1165) | 53.05% (618/1165) | +11.33% | +1.20% |
| MMLU（14042） | 41.61% (5843/14042) | 47.62% (6687/14042) | 42.92% (6027/14042) | +6.01% | +1.31% |

补充信息：

| 项目 | 值 |
|---|---|
| 权重 | `checkpoints/phase2_best` |
| 设备 | `cuda:0` |
| GPU 数 | 2 |
| max_samples | -1 |
| knowledge_miss_count | 全部为 0 |
| ARC skipped | 0 |
| 备注 | `phase2_best`，10 个 epoch，当前实际行为为带独立 `norm` 训练的版本 |
| elapsed_sec | 2221.75s |

## `phase2_epoch4`

来源：
- [results/e2/e2_cross_domain_phase2_epoch4.json](/home/undergraduate/zcy/Explicit-Lora/results/e2/e2_cross_domain_phase2_epoch4.json)

| 数据集 | Baseline | Fusion+知识 | Fusion+空知识 | Δacc | Δacc_empty |
|---|---:|---:|---:|---:|---:|
| MedQA（1273） | 32.91% (419/1273) | 41.24% (525/1273) | 33.54% (427/1273) | +8.33% | +0.63% |
| ARC-Challenge（1165） | 51.85% (604/1165) | 60.94% (710/1165) | 54.08% (630/1165) | +9.10% | +2.23% |
| MMLU（14042） | 41.61% (5843/14042) | 44.25% (6214/14042) | 43.01% (6040/14042) | +2.64% | +1.40% |

补充信息：

| 项目 | 值 |
|---|---|
| 权重 | `checkpoints/phase2_epoch4` |
| 设备 | `cuda:0` |
| GPU 数 | 2 |
| max_samples | -1 |
| knowledge_miss_count | 全部为 0 |
| ARC skipped | 0 |
| elapsed_sec | 2206.09s |

## `p2_qwen3_10ep/phase2_best`（qwen3 10轮版本）

来源：
- [results/e2/e2_cross_domain_p2_qwen3_10ep_phase2_best.json](/home/undergraduate/zcy/Explicit-Lora/results/e2/e2_cross_domain_p2_qwen3_10ep_phase2_best.json)

| 数据集 | Baseline | Fusion+知识 | Fusion+空知识 | Δacc | Δacc_empty |
|---|---:|---:|---:|---:|---:|
| MedQA（1273） | 32.91% (419/1273) | 40.30% (513/1273) | 31.97% (407/1273) | +7.38% | -0.94% |
| ARC-Challenge（1165） | 51.85% (604/1165) | 61.20% (713/1165) | 51.85% (604/1165) | +9.36% | +0.00% |
| MMLU（14042） | 41.61% (5843/14042) | 45.63% (6408/14042) | 41.00% (5757/14042) | +4.02% | -0.61% |

补充信息：

| 项目 | 值 |
|---|---|
| 权重 | `checkpoints/p2_qwen3_10ep/phase2_best` |
| 设备 | `cuda:0` |
| GPU 数 | 2 |
| max_samples | -1 |
| 备注 | `qwen3` 模式，10 个 epoch，复用 Qwen3 encoder 与原始 `norm` |
| knowledge_miss_count | 全部为 0 |
| ARC skipped | 0 |
| elapsed_sec | 2212.92s |

## `e3_fair_compare`

来源：
- [results/e3/e3_fair_compare.json](/home/undergraduate/zcy/Explicit-Lora/results/e3/e3_fair_compare.json)

权重说明：
- `G2 Fusion-Phase1` 使用 `checkpoints/phase2_best`
- `G3 Fusion-Phase2` 使用 `checkpoints/phase3_best`
- 即本次 E3 使用的是两个 `best` 权重

| 组别 | MedQA | ARC | MMLU |
|---|---:|---:|---:|
| G0 Baseline | 32.91% | 51.85% | 41.61% |
| G1 RAG-compressed | 41.56% | 66.52% | 52.50% |
| G2 Fusion-Phase1 | 34.56% | 54.51% | 41.29% |
| G3 Fusion-Phase2 | 67.16% | 72.62% | 60.61% |
| G4 RAG-original | 86.96% | 86.95% | 79.63% |

差值表：

| 指标 | MedQA | ARC | MMLU |
|---|---:|---:|---:|
| G2 - G1 | -6.99% | -12.02% | -11.21% |
| G3 - G1 | +25.61% | +6.09% | +8.11% |
| G4 - G0 | +54.05% | +35.11% | +38.02% |

效率表：

| 指标 | MedQA | ARC | MMLU |
|---|---:|---:|---:|
| eff_G2 | 3.1% | 7.6% | -0.8% |
| eff_G3 | 63.4% | 59.2% | 50.0% |

补充信息：

| 项目 | 值 |
|---|---|
| 输出文件 | `results/e3/e3_fair_compare.json` |
| G2 权重 | `checkpoints/phase2_best` |
| G3 权重 | `checkpoints/phase3_best` |
| elapsed_sec | 3467.81s |
