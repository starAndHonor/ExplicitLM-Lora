# E4 SFT Ablation

This directory hosts a local migration of the reference E4 experiment using the
current repository's code only.

It compares three groups on `medqa`, `arc`, and `mmlu`:

- `Baseline`
- `Phase1`
- `Phase2`

The key derived metric is:

- `sft_cost = phase1_acc - phase2_acc`

Positive `sft_cost` means the SFT stage hurt cross-domain performance relative
to the pre-SFT fusion checkpoint.

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e4/run_e4.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase2_best \
  --phase2-weights checkpoints/phase3_best \
  --device cuda:0
```
