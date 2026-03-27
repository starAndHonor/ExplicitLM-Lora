# E2 Ref

This directory is copied from `Reference/Explicit-Lora-fusion/evaluation/`
and minimally adapted so the Reference-style E2 evaluation can run against the
current repository's model/checkpoint layout.

Run:

```bash
conda run -n ExplicitLLM python experiments/e2-ref/run_e2_ref.py \
  --config config/default.yaml \
  --fusion-ckpt checkpoints/phase2_best
```
