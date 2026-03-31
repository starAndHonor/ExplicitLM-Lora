# E2 Cross-Domain Experiment

This directory now hosts a local E2 implementation adapted from the reference
evaluation pipeline, but rewritten to use the current repository's code and
configuration.

The experiment evaluates three conditions on `medqa`, `arc`, and `mmlu` for
both Phase 2 and Phase 3 checkpoints:

- `baseline`
- `phase2.fusion_knowledge`
- `phase2.fusion_empty`
- `phase3.fusion_knowledge`
- `phase3.fusion_empty`

It writes a report under `results/e2/` with:

- per-dataset `phase2` results
- per-dataset `phase3` results
- `phase3_vs_phase2`

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```
