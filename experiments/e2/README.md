# E2 Cross-Domain Experiment

This directory now hosts a local E2 implementation adapted from the reference
evaluation pipeline, but rewritten to use the current repository's code and
configuration.

The experiment evaluates `baseline`, `Phase 2 fusion`, and `Phase 3 fusion` on
`medqa`, `arc`, and `mmlu`.

It writes a report under `results/e2/` with:

- a `Phase 2` result block
- a `Phase 3` result block
- a `Phase3 - Phase2` comparison block

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```
