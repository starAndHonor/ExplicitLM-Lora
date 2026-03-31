# E3 Fair Compare

This directory hosts a local migration of the reference E3 experiment using the
current repository's code only.

It evaluates five groups on `medqa`, `arc`, and `mmlu`:

- `G0_baseline`
- `G1_rag_compressed`
- `G2_fusion_phase1`
- `G3_fusion_phase2`
- `G4_rag_original`

By default it runs with `k=64`. You can also set `k=32/64/128/256` to switch
knowledge token budgets and corresponding `*_knowledge_k*.jsonl` files.

Example (`k=64`):

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```

Example (`k=128`):

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase1_best \
  --phase2-weights checkpoints/phase2_best \
  --k 128 \
  --device cuda:0
```

Multi-k helper script:

```bash
ENC_MODE=qwen3 \
GPU_IDS=2,3 \
PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_e3_multik.sh
```

When multiple `E3` result files with different `k` exist under `results/e3/`,
`scripts/collect_results.py` will generate an additional multi-k summary block
in `results/results_summary.md`.
