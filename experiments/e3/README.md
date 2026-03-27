# E3 Fair Compare

This directory hosts a local migration of the reference E3 experiment using the
current repository's code only.

It evaluates five groups on `medqa`, `arc`, and `mmlu`:

- `G0_baseline`
- `G1_rag_compressed`
- `G2_fusion_phase1`
- `G3_fusion_phase2`
- `G4_rag_original`

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase1_best \
  --phase2-weights checkpoints/phase2_best \
  --device cuda:0
```
