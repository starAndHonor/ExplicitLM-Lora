# E4 SFT Ablation

This directory hosts a local migration of the reference E4 experiment using the
current repository's code only.

It compares three groups on `medqa`, `arc`, and `mmlu`:

- `Baseline`
- `Phase2`
- `Phase3`

The key derived metric is:

- `sft_effect = phase3_acc - phase2_acc`

Positive `sft_effect` means the SFT stage improved cross-domain performance
relative to the pre-SFT fusion checkpoint.

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e4/run_e4.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```
