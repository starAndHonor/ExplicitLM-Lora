# E1 Sanity Check

This directory hosts a local E1 implementation adapted from the reference
`counterfactual_eval.py`, but wired into the current repository's model loader.

The experiment evaluates three conditions on MedQA test:

- `correct knowledge`
- `counterfactual knowledge`
- `no knowledge` (all-pad knowledge ids)

Metric:

- `KS = acc_correct - acc_counterfactual`

Notes:

- This implementation follows the reference E1 semantics: all three groups use
  the same fusion model, and the `no knowledge` group is implemented as
  all-pad injected knowledge rather than the raw baseline LM.
- Counterfactual knowledge is auto-built to
  `data/medqa_knowledge_counterfactual.jsonl` if missing.

Example:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e1/run_e1.py \
  --config config/default.yaml \
  --weights checkpoints/phase2_best
```

When driven by `scripts/run_experiment*.sh`, `e1` is executed twice by default:

- once with `PHASE2_WEIGHTS`
- once with `PHASE3_WEIGHTS`

This produces two result JSON files that are later grouped into the E1
comparison table in `results/results_summary.md`.

Current observed results:

- `checkpoints/p2_qwen3_10ep/phase2_best`
  - `acc_correct = 43.44%`
  - `acc_counterfactual = 29.07%`
  - `acc_no_knowledge = 31.97%`
  - `KS = +14.38%`

- `checkpoints/p3_from_p2_qwen3_10ep/phase3_best`
  - `acc_correct = 70.07%`
  - `acc_counterfactual = 13.67%`
  - `acc_no_knowledge = 31.58%`
  - `KS = +56.40%`

Summary:

- `Phase 3 qwen3` improves `KS` from `14.38%` to `56.40%`, showing much stronger dependence on injected knowledge after SFT.
