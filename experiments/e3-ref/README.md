# E3 Reference-Compatible Eval

This directory contains a copied-and-adapted version of the Reference E3 fair
comparison pipeline.

Run it with:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e3-ref/run_e3_ref.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase1_best \
  --phase2-weights checkpoints/phase2_best \
  --tag phase2_ref_gpu23
```

The copied Reference code is adapted to:

- use the current repo config
- load current checkpoints through `model_compat.py`
- keep Reference-style reporting and output naming
