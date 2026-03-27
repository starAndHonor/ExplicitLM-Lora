# E2 Cross-Domain Experiment

This directory now hosts a local E2 implementation adapted from the reference
evaluation pipeline, but rewritten to use the current repository's code and
configuration.

The experiment evaluates three conditions on `medqa`, `arc`, and `mmlu`:

- `baseline`
- `fusion_knowledge`
- `fusion_empty`

It writes a report under `results/e2/` with both `delta_acc` and
`delta_acc_empty`.

Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --fusion-ckpt checkpoints/phase2_best \
  --device cuda:0
```
