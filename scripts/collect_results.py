from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "results_summary.md"

sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config


@dataclass
class Section:
    title: str
    lines: list[str]


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _ckpt_label(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "-"
    path = Path(value)
    parts = path.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        tail = parts[idx + 1 :]
        return "/".join(tail) if tail else path.name
    return path.name or value


def _pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "-"


def _signed_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:+.2f}%"
    return "-"


def _num(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def _is_result_json(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if any(part.startswith("_tmp") for part in path.parts):
        return False
    if path.name.startswith("_"):
        return False
    return True


def _scan_results() -> list[Path]:
    return sorted(path for path in RESULTS_DIR.rglob("*.json") if _is_result_json(path))


def _model_overview() -> list[str]:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    model = cfg.model
    router = cfg.router
    train = cfg.train
    data = cfg.data
    return [
        "## Model Config",
        "",
        "- `base_model`: `{}`".format(model.base_model),
        "- `hidden_dim`: `{}`".format(model.hidden_dim),
        "- `num_layers`: `{}`".format(model.num_layers),
        "- `injection_method`: `{}`".format(model.injection_method),
        "- `injection_layers`: `{}`".format(model.injection_layers),
        "- `encoder_depth`: `{}`".format(model.encoder_depth),
        "- `retrieval_encoder_depth`: `{}`".format(getattr(model, "retrieval_encoder_depth", model.encoder_depth)),
        "- `fusion_encoder_depth`: `{}`".format(getattr(model, "fusion_encoder_depth", model.encoder_depth)),
        "- `knowledge_encoder_mode`: `{}`".format(model.knowledge_encoder_mode),
        "- `fusion_length`: `{}`".format(model.fusion_length),
        "- `anchor_length`: `{}`".format(model.anchor_length),
        "- `router.num_candidates`: `{}`".format(router.num_candidates),
        "- `router.temperature`: `{}`".format(router.temperature),
        "- `train.phase2_max_epochs`: `{}`".format(train.phase2_max_epochs),
        "- `train.phase3_max_epochs`: `{}`".format(train.phase3_max_epochs),
        "- `data.phase2_n_samples_per_epoch`: `{}`".format(data.phase2_n_samples_per_epoch),
        "- `data.phase3_max_seq_length`: `{}`".format(data.phase3_max_seq_length),
        "",
    ]


def _table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    rows = list(rows)
    if not rows:
        return []
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _config_summary(data: dict[str, Any]) -> list[str]:
    items: list[tuple[str, str]] = []
    field_map = [
        ("weights", "Weights"),
        ("phase1_weights", "Phase 2 Weights"),
        ("phase2_weights", "Phase 3 Weights"),
        ("phase3_weights", "Phase 3 Weights"),
        ("k", "Knowledge Budget"),
        ("device", "Device"),
        ("num_gpus", "Num GPUs"),
        ("max_samples", "Max Samples"),
        ("n_warmup", "Warmup"),
        ("n_measure", "Measure"),
        ("elapsed_sec", "Elapsed Sec"),
    ]
    for key, label in field_map:
        value = data.get(key)
        if value is None:
            continue
        if key.endswith("weights") or key == "weights":
            rendered = _ckpt_label(value)
        elif key == "elapsed_sec":
            rendered = _num(value, 2)
        else:
            rendered = str(value)
        items.append((label, rendered))
    if not items:
        return []
    return ["**Config**", ""] + [f"- `{label}`: `{value}`" for label, value in items] + [""]


def _summarize_e1(path: Path, data: dict[str, Any]) -> Section:
    rows = [[
        _rel(path),
        _ckpt_label(data.get("weights")),
        _pct(data.get("acc_correct")),
        _pct(data.get("acc_counterfactual")),
        _pct(data.get("acc_no_knowledge")),
        _pct(data.get("knowledge_sensitivity")),
        str(data.get("total", "-")),
    ]]
    return Section(
        "E1",
        _config_summary(data)
        + _table(
            ["File", "Weights", "Correct", "Counterfactual", "No Knowledge", "KS", "Total"],
            rows,
        ),
    )


def _format_ratio(acc: Any, correct: Any, total: Any) -> str:
    if isinstance(acc, (int, float)) and isinstance(correct, int) and isinstance(total, int):
        return f"{acc * 100:.2f}% ({correct}/{total})"
    return "-"


def _summarize_e1_group(paths: list[Path]) -> Section | None:
    grouped: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight = _ckpt_label(data.get("weights"))
        grouped[weight] = (path, data)
    if not grouped:
        return None

    ordered_items = sorted(grouped.items(), key=lambda item: item[0])
    if len(ordered_items) == 2:
        headers = ["指标", "Phase 1 权重", "Phase 2 权重"]
    else:
        headers = ["指标"] + [weight for weight, _ in ordered_items]
    totals = [entry[1].get("total", "-") for _, entry in ordered_items]
    devices = [str(entry[1].get("device", "-")) for _, entry in ordered_items]
    files = [_rel(entry[0]) for _, entry in ordered_items]

    config_lines = [
        "**Config**",
        "",
        f"- `Files`: `{files}`",
        f"- `Devices`: `{devices}`",
        f"- `Totals`: `{totals}`",
    ]
    if len(ordered_items) == 2:
        config_lines.extend([
            f"- `Phase 1 权重`: `{ordered_items[0][0]}`",
            f"- `Phase 2 权重`: `{ordered_items[1][0]}`",
        ])
    config_lines.append("")

    rows = [
        ["acc_correct（正确知识）"]
        + [
            _format_ratio(data.get("acc_correct"), data.get("correct_correct"), data.get("total"))
            for _, (_, data) in ordered_items
        ],
        ["acc_counterfactual（反事实知识）"]
        + [
            _format_ratio(
                data.get("acc_counterfactual"),
                data.get("correct_counterfactual"),
                data.get("total"),
            )
            for _, (_, data) in ordered_items
        ],
        ["acc_no_knowledge（全 pad）"]
        + [
            _format_ratio(
                data.get("acc_no_knowledge"),
                data.get("correct_no_knowledge"),
                data.get("total"),
            )
            for _, (_, data) in ordered_items
        ],
        ["KS"]
        + [
            ("+" if isinstance(data.get("knowledge_sensitivity"), (int, float)) and data.get("knowledge_sensitivity") >= 0 else "")
            + _pct(data.get("knowledge_sensitivity"))
            for _, (_, data) in ordered_items
        ],
    ]

    lines = config_lines + _table(headers, rows)
    return Section("E1", lines)


def _summarize_e2(path: Path, data: dict[str, Any]) -> Section:
    lines = _config_summary(data)

    def ds_label(ds: str, total: Any) -> str:
        if ds == "medqa":
            return f"MedQA（{total:,} 题）" if isinstance(total, int) else "MedQA"
        if ds == "arc":
            return f"ARC-Challenge（{total:,} 题）" if isinstance(total, int) else "ARC-Challenge"
        return f"MMLU（{total:,} 题）" if isinstance(total, int) else "MMLU"

    has_multi_phase = any(isinstance(data.get(ds, {}).get("phase2"), dict) for ds in ("medqa", "arc", "mmlu"))
    if not has_multi_phase:
        rows: list[list[str]] = []
        for ds in ("medqa", "arc", "mmlu"):
            ds_data = data.get(ds, {})
            total = ds_data.get("baseline", {}).get("total", "-")
            rows.append([
                ds_label(ds, total),
                _pct(ds_data.get("baseline", {}).get("acc")),
                _pct(ds_data.get("fusion_knowledge", {}).get("acc")),
                _pct(ds_data.get("fusion_empty", {}).get("acc")),
                _signed_pct(ds_data.get("delta_acc")),
            ])
        lines.extend(_table(["数据集", "Baseline", "Fusion+知识", "Fusion+空知识", "Δacc"], rows))
        return Section("E2", lines)

    lines.extend(["**Phase 2**", ""])
    phase2_rows: list[list[str]] = []
    phase3_rows: list[list[str]] = []
    compare_rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = data.get(ds, {})
        total = ds_data.get("baseline", {}).get("total", "-")
        label = ds_label(ds, total)
        baseline = _pct(ds_data.get("baseline", {}).get("acc"))
        phase2 = ds_data.get("phase2", {})
        phase3 = ds_data.get("phase3", {})
        phase2_rows.append([
            label,
            baseline,
            _pct(phase2.get("fusion_knowledge", {}).get("acc")),
            _pct(phase2.get("fusion_empty", {}).get("acc")),
            _signed_pct(phase2.get("delta_acc")),
        ])
        phase3_rows.append([
            label,
            baseline,
            _pct(phase3.get("fusion_knowledge", {}).get("acc")),
            _pct(phase3.get("fusion_empty", {}).get("acc")),
            _signed_pct(phase3.get("delta_acc")),
        ])
        compare_rows.append([
            label,
            _signed_pct(phase2.get("delta_acc")),
            _signed_pct(phase3.get("delta_acc")),
            _signed_pct(ds_data.get("phase3_vs_phase2")),
        ])
    lines.extend(_table(["数据集", "Baseline", "Fusion+知识", "Fusion+空知识", "Δacc"], phase2_rows))
    lines.extend(["", "**Phase 3**", ""])
    lines.extend(_table(["数据集", "Baseline", "Fusion+知识", "Fusion+空知识", "Δacc"], phase3_rows))
    lines.extend(["", "**Phase 3 vs Phase 2**", ""])
    lines.extend(_table(["数据集", "Phase2 Δacc", "Phase3 Δacc", "Phase3-Phase2"], compare_rows))
    return Section("E2", lines)


def _summarize_e3(path: Path, data: dict[str, Any]) -> Section:
    medqa = data.get("medqa", {})
    arc = data.get("arc", {})
    mmlu = data.get("mmlu", {})
    rows_main = [
        ["G0 Baseline", _pct(medqa.get("G0_baseline", {}).get("acc")), _pct(arc.get("G0_baseline", {}).get("acc")), _pct(mmlu.get("G0_baseline", {}).get("acc"))],
        ["G1 RAG-compressed", _pct(medqa.get("G1_rag_compressed", {}).get("acc")), _pct(arc.get("G1_rag_compressed", {}).get("acc")), _pct(mmlu.get("G1_rag_compressed", {}).get("acc"))],
        ["G2 Fusion-Phase1", _pct(medqa.get("G2_fusion_phase1", {}).get("acc")), _pct(arc.get("G2_fusion_phase1", {}).get("acc")), _pct(mmlu.get("G2_fusion_phase1", {}).get("acc"))],
        ["G3 Fusion-Phase2", _pct(medqa.get("G3_fusion_phase2", {}).get("acc")), _pct(arc.get("G3_fusion_phase2", {}).get("acc")), _pct(mmlu.get("G3_fusion_phase2", {}).get("acc"))],
        ["G4 RAG-original", _pct(medqa.get("G4_rag_original", {}).get("acc")), _pct(arc.get("G4_rag_original", {}).get("acc")), _pct(mmlu.get("G4_rag_original", {}).get("acc"))],
    ]
    summary = data.get("summary", {})
    rows_eff = [
        ["MedQA", _pct(summary.get("medqa", {}).get("efficiency_G2")), _pct(summary.get("medqa", {}).get("efficiency_G3"))],
        ["ARC", _pct(summary.get("arc", {}).get("efficiency_G2")), _pct(summary.get("arc", {}).get("efficiency_G3"))],
        ["MMLU", _pct(summary.get("mmlu", {}).get("efficiency_G2")), _pct(summary.get("mmlu", {}).get("efficiency_G3"))],
    ]
    return Section(
        "E3",
        _config_summary(data)
        + _table(["组别", "MedQA", "ARC", "MMLU"], rows_main)
        + ["", "*Efficiency*", ""]
        + _table(["数据集", "G2 效率", "G3 效率"], rows_eff),
    )


def _summarize_e3_multik(paths: list[Path]) -> Section | None:
    entries: list[tuple[int, Path, dict[str, Any]]] = []
    phase1_weight = None
    phase2_weight = None
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        k = data.get("k")
        if not isinstance(k, int):
            continue
        if phase1_weight is None:
            phase1_weight = _ckpt_label(data.get("phase1_weights"))
            phase2_weight = _ckpt_label(data.get("phase2_weights"))
        entries.append((k, path, data))
    if len(entries) < 2:
        return None
    entries.sort(key=lambda item: item[0])

    lines = [
        "**Config**",
        "",
        f"- `Phase1 Weights`: `{phase1_weight}`",
        f"- `Phase2 Weights`: `{phase2_weight}`",
        f"- `Knowledge Budgets`: `{[k for k, _, _ in entries]}`",
        "",
        "**Multi-k Summary**",
        "",
    ]

    for ds_name, ds_label in (("medqa", "MedQA"), ("arc", "ARC"), ("mmlu", "MMLU")):
        rows_main: list[list[str]] = []
        rows_eff: list[list[str]] = []
        for k, _, data in entries:
            ds = data.get(ds_name, {})
            summary = data.get("summary", {}).get(ds_name, {})
            rows_main.append([
                str(k),
                _pct(ds.get("G1_rag_compressed", {}).get("acc")),
                _pct(ds.get("G2_fusion_phase1", {}).get("acc")),
                _pct(ds.get("G3_fusion_phase2", {}).get("acc")),
                _pct(ds.get("G4_rag_original", {}).get("acc")),
            ])
            rows_eff.append([
                str(k),
                _pct(summary.get("efficiency_G2")),
                _pct(summary.get("efficiency_G3")),
            ])
        lines.append(f"**{ds_label}**")
        lines.append("")
        lines.extend(_table(["k", "G1 RAG-compressed", "G2 Fusion-Phase1", "G3 Fusion-Phase2", "G4 RAG-original"], rows_main))
        lines.extend(["", "*Efficiency*", ""])
        lines.extend(_table(["k", "G2 效率", "G3 效率"], rows_eff))
        lines.append("")

    return Section("E3", lines)


def _summarize_e4(path: Path, data: dict[str, Any]) -> Section:
    rows: list[list[str]] = []
    ablation = data.get("ablation", {})
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = ablation.get(ds, {})
        sft_effect = ds_data.get("sft_effect")
        if sft_effect is None and ds_data.get("sft_cost") is not None:
            sft_effect = -ds_data.get("sft_cost")
        if ds == "medqa":
            ds_label = "MedQA"
        elif ds == "arc":
            ds_label = "ARC"
        else:
            ds_label = "MMLU"
        rows.append([
            ds_label,
            _pct(ds_data.get("baseline_acc")),
            _pct(ds_data.get("phase1_acc")),
            _pct(ds_data.get("phase2_acc")),
            ("+" if isinstance(ds_data.get("phase1_delta"), (int, float)) and ds_data.get("phase1_delta") >= 0 else "") + _pct(ds_data.get("phase1_delta")),
            ("+" if isinstance(ds_data.get("phase2_delta"), (int, float)) and ds_data.get("phase2_delta") >= 0 else "") + _pct(ds_data.get("phase2_delta")),
            ("+" if isinstance(sft_effect, (int, float)) and sft_effect >= 0 else "") + _pct(sft_effect),
        ])
    return Section(
        "E4",
        _config_summary(data)
        + _table(
            ["数据集", "Baseline", "Phase 1", "Phase 2", "Phase1 Δ", "Phase2 Δ", "SFT 效果"],
            rows,
        ),
    )


def _summarize_e5(path: Path, data: dict[str, Any]) -> Section:
    e5a = data.get("e5a", {})
    e5b = data.get("e5b", {})
    dataset_titles = {
        "medqa": "MedQA",
        "arc": "ARC",
        "mmlu": "MMLU",
    }
    token_budgets = (32, 64, 128, 256)
    lines = _config_summary(data)
    lines.append("**E5-A：知识 Token 预算分析**")
    lines.append("")
    for ds in ("medqa", "arc", "mmlu"):
        a = e5a.get(ds, {})
        rows: list[list[str]] = []
        for k in token_budgets:
            rag_acc = a.get(f"rag_k{k}", {}).get("acc")
            fusion_acc = a.get(f"fusion_p2_k{k}", {}).get("acc")
            rows.append([
                str(k),
                _pct(a.get("baseline", {}).get("acc")),
                _pct(fusion_acc),
                _pct(rag_acc),
                _signed_pct(
                    fusion_acc - rag_acc
                    if isinstance(fusion_acc, (int, float)) and isinstance(rag_acc, (int, float))
                    else None
                ),
            ])
        lines.append(f"**{dataset_titles[ds]}**")
        lines.append("")
        lines.extend(
            _table(
                ["Token", "Baseline", "Fusion-P2", "RAG", "Δ(Fusion-RAG)"],
                rows,
            )
        )
        lines.append("")

    lines.append("**E5-B：知识相关性分析（k=64）**")
    lines.append("")
    lines.extend(
        _table(
            ["条件", "P1 MedQA", "P2 MedQA", "P1 ARC", "P2 ARC", "P1 MMLU", "P2 MMLU"],
            [
                [
                    "Oracle",
                    _pct(e5b.get("medqa", {}).get("oracle_p1", {}).get("acc")),
                    _pct(e5b.get("medqa", {}).get("oracle_p2", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("oracle_p1", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("oracle_p2", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("oracle_p1", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("oracle_p2", {}).get("acc")),
                ],
                [
                    "Shuffled",
                    _pct(e5b.get("medqa", {}).get("shuffled_p1", {}).get("acc")),
                    _pct(e5b.get("medqa", {}).get("shuffled_p2", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("shuffled_p1", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("shuffled_p2", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("shuffled_p1", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("shuffled_p2", {}).get("acc")),
                ],
                [
                    "Empty",
                    _pct(e5b.get("medqa", {}).get("empty_p1", {}).get("acc")),
                    _pct(e5b.get("medqa", {}).get("empty_p2", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("empty_p1", {}).get("acc")),
                    _pct(e5b.get("arc", {}).get("empty_p2", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("empty_p1", {}).get("acc")),
                    _pct(e5b.get("mmlu", {}).get("empty_p2", {}).get("acc")),
                ],
            ],
        )
    )
    return Section("E5", lines)


def _summarize_e6(path: Path, data: dict[str, Any]) -> Section:
    benches = data.get("benchmarks", {})
    acc = data.get("accuracy_data", {}).get("medqa", {})
    rows = []
    label_map = {
        "baseline": "Baseline",
        "rag_compressed": "RAG-compressed@64",
        "fusion": "Fusion-Phase2@64",
        "rag_original": "RAG-original@~256",
    }
    context_map = {
        "baseline": "0 tokens",
        "rag_compressed": "~64 tokens",
        "fusion": "0 tokens",
        "rag_original": "~256 tokens",
    }
    for key in ("baseline", "rag_compressed", "fusion", "rag_original"):
        b = benches.get(key, {})
        rows.append([
            label_map.get(key, b.get("label", key)),
            _num(b.get("latency_ms")),
            _num(b.get("throughput")),
            _num(b.get("peak_memory_mb"), 1),
            _num(b.get("avg_input_len"), 1),
            context_map.get(key, str(b.get("context_tokens", "-"))),
        ])
    lines = _config_summary(data) + _table(
        ["方法", "延迟 (ms/样本)", "吞吐 (样本/s)", "显存 (MB)", "平均输入长度", "上下文占用"],
        rows,
    )
    rag_abs_acc = acc.get("rag_original", acc.get("rag_k256"))
    fusion_abs_acc = acc.get("fusion_phase2", acc.get("fusion_p2_k64"))
    rag_acc64 = acc.get("rag_k64")
    fusion_acc64 = acc.get("fusion_p2_k64")
    rag_b = benches.get("rag_original")
    fusion_b = benches.get("fusion")
    if (
        isinstance(rag_abs_acc, (int, float))
        and isinstance(fusion_abs_acc, (int, float))
        and isinstance(rag_b, dict)
        and isinstance(fusion_b, dict)
    ):
        latency_delta = None
        if isinstance(rag_b.get("latency_ms"), (int, float)) and rag_b.get("latency_ms"):
            latency_delta = (fusion_b.get("latency_ms") - rag_b.get("latency_ms")) / rag_b.get("latency_ms")
        memory_delta = None
        if isinstance(rag_b.get("peak_memory_mb"), (int, float)) and rag_b.get("peak_memory_mb"):
            memory_delta = (fusion_b.get("peak_memory_mb") - rag_b.get("peak_memory_mb")) / rag_b.get("peak_memory_mb")
        lines.extend([
            "",
            "**修正后的六维对比汇总表（E3 准确率 + E6 效率）**",
            "",
            * _table(
                ["维度", "RAG-original (G4)", "Fusion Phase 2 (G3)", "胜者"],
                [
                    [
                        "绝对准确率",
                        f"{_pct(rag_abs_acc)} (MedQA)",
                        f"{_pct(fusion_abs_acc)} (MedQA)",
                        "RAG" if rag_abs_acc > fusion_abs_acc else "Fusion",
                    ],
                    [
                        "同 token 准确率(k=64)",
                        _pct(rag_acc64),
                        (
                            f"{_pct(fusion_acc64)} ({_signed_pct(fusion_acc64 - rag_acc64)})"
                            if isinstance(rag_acc64, (int, float)) and isinstance(fusion_acc64, (int, float))
                            else _pct(fusion_acc64)
                        ),
                        "Fusion" if isinstance(rag_acc64, (int, float)) and isinstance(fusion_acc64, (int, float)) and fusion_acc64 > rag_acc64 else "RAG",
                    ],
                    [
                        "推理延迟",
                        f"{_num(rag_b.get('latency_ms'))} ms",
                        (
                            f"{_num(fusion_b.get('latency_ms'))} ms ({_signed_pct(latency_delta)})"
                            if isinstance(latency_delta, (int, float))
                            else f"{_num(fusion_b.get('latency_ms'))} ms"
                        ),
                        "Fusion" if isinstance(fusion_b.get("latency_ms"), (int, float)) and isinstance(rag_b.get("latency_ms"), (int, float)) and fusion_b.get("latency_ms") < rag_b.get("latency_ms") else "RAG",
                    ],
                    [
                        "峰值显存",
                        f"{_num(rag_b.get('peak_memory_mb'), 1)} MB",
                        (
                            f"{_num(fusion_b.get('peak_memory_mb'), 1)} MB ({_signed_pct(memory_delta)})"
                            if isinstance(memory_delta, (int, float))
                            else f"{_num(fusion_b.get('peak_memory_mb'), 1)} MB"
                        ),
                        "Fusion" if isinstance(fusion_b.get("peak_memory_mb"), (int, float)) and isinstance(rag_b.get("peak_memory_mb"), (int, float)) and fusion_b.get("peak_memory_mb") < rag_b.get("peak_memory_mb") else "RAG",
                    ],
                    [
                        "上下文窗口侵占",
                        "~256 token",
                        "0 token",
                        "Fusion",
                    ],
                    [
                        "知识可预编码缓存",
                        "否（每次重处理）",
                        "是（编码一次复用）",
                        "Fusion",
                    ],
                ],
            ),
        ])
    return Section("E6", lines)


def _summarize_e7(path: Path, data: dict[str, Any]) -> Section:
    lines = [
        "**Config**",
        "",
        f"- `Training-free Weights`: `{_ckpt_label(data.get('training_free_weights'))}`",
        f"- `Device`: `{data.get('device', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Max Samples`: `{data.get('max_samples', '-')}`",
        f"- `Elapsed Sec`: `{_num(data.get('elapsed_sec'), 2)}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_title = "MedQA" if ds == "medqa" else "ARC" if ds == "arc" else "MMLU"
        ds_data = data.get(ds, {})
        summary = data.get("summary", {}).get(ds, {})
        rows.append([
            ds_title,
            _pct(ds_data.get("B0_qwen3_base", {}).get("acc")),
            _pct(ds_data.get("TF_dense_p3_infer", {}).get("acc")),
            _pct(ds_data.get("RAG_dense", {}).get("acc")),
            _signed_pct(summary.get("TF_minus_B0")),
            _signed_pct(summary.get("RAG_minus_B0")),
            str(summary.get("best_group", "-")),
        ])
    lines.extend(
        _table(
            ["数据集", "B0", "TF Dense→P3", "Dense RAG", "TF-B0", "RAG-B0", "Best"],
            rows,
        )
    )
    return Section("E7", lines)


def _summarize_e7_retrieval(path: Path, data: dict[str, Any]) -> Section:
    query_mode = "-"
    rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_title = "MedQA" if ds == "medqa" else "ARC" if ds == "arc" else "MMLU"
        ds_data = data.get(ds, {})
        results = ds_data.get("results", {})
        if results:
            query_mode = next(iter(results.keys()), query_mode)
        result = results.get(query_mode, {}) if query_mode in results else (next(iter(results.values())) if results else {})
        rows.append([
            ds_title,
            str(result.get("tested", "-")),
            _pct(result.get("top1_exact_rate")),
            _pct(result.get("topk_hit_rate")),
            str(result.get("top_k", "-")),
        ])
    lines = [
        "**Config**",
        "",
        f"- `Query Mode`: `{query_mode}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    lines.extend(_table(["数据集", "Tested", "Top1", "Top16", "k"], rows))
    return Section("E7", lines)


def _summarize_e8a(path: Path, data: dict[str, Any]) -> Section:
    metrics = data.get("metrics", {})
    lines = [
        "**Config**",
        "",
        f"- `Dataset`: `{data.get('dataset', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Edits`: `{data.get('n_edits', '-')}`",
        f"- `Seed`: `{data.get('seed', '-')}`",
        f"- `Phase3 Weights`: `{_ckpt_label(data.get('phase3_weights'))}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    lines.extend(
        _table(
            ["指标", "数值"],
            [
                ["Pre-write Acc", _pct(metrics.get("pre_write_acc"))],
                ["Post-write Acc", _pct(metrics.get("post_write_acc"))],
                ["Write Success Rate", _pct(metrics.get("write_success_rate"))],
                ["Retrieval Top1 Before", _pct(metrics.get("retrieval_top1_before"))],
                ["Retrieval Top1 After", _pct(metrics.get("retrieval_top1_after"))],
                ["Mean Write Latency (ms)", _num(metrics.get("mean_write_latency_ms"), 2)],
            ],
        )
    )
    return Section("E8", lines)


def _summarize_e8b(path: Path, data: dict[str, Any]) -> Section:
    metrics = data.get("metrics", {})
    lines = [
        "**Config**",
        "",
        f"- `Dataset`: `{data.get('dataset', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Cases`: `{data.get('n_cases', '-')}`",
        f"- `Seed`: `{data.get('seed', '-')}`",
        f"- `Phase3 Weights`: `{_ckpt_label(data.get('phase3_weights'))}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    lines.extend(
        _table(
            ["指标", "数值"],
            [
                ["Pre-delete Acc", _pct(metrics.get("pre_delete_acc"))],
                ["Post-delete Acc", _pct(metrics.get("post_delete_acc"))],
                ["Post-rollback Acc", _pct(metrics.get("post_rollback_acc"))],
                ["Delete Success Rate", _pct(metrics.get("delete_success_rate"))],
                ["Rollback Fidelity", _pct(metrics.get("rollback_fidelity"))],
                ["Retrieval Top1 Before", _pct(metrics.get("retrieval_top1_before"))],
                ["Retrieval Top1 After Delete", _pct(metrics.get("retrieval_top1_after_delete"))],
                ["Retrieval Top1 After Rollback", _pct(metrics.get("retrieval_top1_after_rollback"))],
                ["Mean Delete Latency (ms)", _num(metrics.get("mean_delete_latency_ms"), 2)],
                ["Mean Rollback Latency (ms)", _num(metrics.get("mean_rollback_latency_ms"), 2)],
            ],
        )
    )
    return Section("E8", lines)


def _summarize_e8c(path: Path, data: dict[str, Any]) -> Section:
    baseline = data.get("baseline", {})
    step_details = data.get("step_details", [])
    lines = [
        "**Config**",
        "",
        f"- `Dataset`: `{data.get('dataset', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Steps`: `{data.get('steps', '-')}`",
        f"- `Seed`: `{data.get('seed', '-')}`",
        f"- `Locality Samples`: `{data.get('locality_samples', '-')}`",
        f"- `Operation Pattern`: `{data.get('operation_pattern', '-')}`",
        f"- `Phase3 Weights`: `{_ckpt_label(data.get('phase3_weights'))}`",
        f"- `File`: `{_rel(path)}`",
        "",
        "**Baseline**",
        "",
    ]
    lines.extend(
        _table(
            ["指标", "数值"],
            [
                ["Full Edit QA Acc", _pct(baseline.get("full_edit_qa_acc"))],
                ["Full Edit Retrieval Top1", _pct(baseline.get("full_edit_retrieval_top1"))],
                ["Locality QA Acc", _pct(baseline.get("locality_qa_acc"))],
                ["Locality Retrieval Top1", _pct(baseline.get("locality_retrieval_top1"))],
            ],
        )
    )
    if step_details:
        lines.extend([
            "",
            "**Key Steps**",
            "",
        ])
        key_rows: list[list[str]] = []
        for detail in step_details:
            last_op = detail.get("last_operation", {})
            key_rows.append([
                str(detail.get("step", "-")),
                str(last_op.get("op", "-")),
                _pct(detail.get("present_qa_acc")),
                _pct(detail.get("present_retrieval_top1")),
                _pct(detail.get("delete_success_rate")),
                _pct(detail.get("rollback_fidelity")),
                _pct(detail.get("locality_retention")),
            ])
        lines.extend(
            _table(
                ["Step", "Last Op", "Edit Acc", "Edit Top1", "Delete Success", "Rollback Fidelity", "Locality Retention"],
                key_rows,
            )
        )
    return Section("E8", lines)


def _summarize_e8d_a(path: Path, data: dict[str, Any]) -> Section:
    metrics = data.get("metrics", {})
    lines = [
        "**Config**",
        "",
        f"- `Dataset`: `{data.get('dataset', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Ingested`: `{data.get('n_ingested', '-')}`",
        f"- `Locality Samples`: `{data.get('locality_samples', '-')}`",
        f"- `Seed`: `{data.get('seed', '-')}`",
        f"- `Phase3 Weights`: `{_ckpt_label(data.get('phase3_weights'))}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    lines.extend(
        _table(
            ["指标", "数值"],
            [
                ["Old QA Retention", _pct(metrics.get("old_qa_retention"))],
                ["Old Retrieval Retention", _pct(metrics.get("old_retrieval_retention"))],
                ["Ingest QA Acc Before", _pct(metrics.get("ingest_qa_acc_before"))],
                ["Ingest QA Acc After", _pct(metrics.get("ingest_qa_acc_after"))],
                ["Ingest Retrieval Top1 Before", _pct(metrics.get("ingest_retrieval_top1_before"))],
                ["Ingest Retrieval Top1 After", _pct(metrics.get("ingest_retrieval_top1_after"))],
                ["Bulk Ingest Latency (ms)", _num(metrics.get("bulk_ingest_latency_ms"), 2)],
                ["Mean Ingest Latency (ms)", _num(metrics.get("mean_ingest_latency_ms"), 2)],
                ["Throughput (docs/s)", _num(metrics.get("ingest_throughput_docs_per_sec"), 2)],
            ],
        )
    )
    return Section("E8", lines)


def _summarize_e8d_b(path: Path, data: dict[str, Any]) -> Section:
    metrics = data.get("metrics", {})
    add_stage = data.get("add_stage", [])
    delete_stage = data.get("delete_stage", [])
    lines = [
        "**Config**",
        "",
        f"- `Dataset`: `{data.get('dataset', '-')}`",
        f"- `Query Mode`: `{data.get('query_mode', '-')}`",
        f"- `Add Count`: `{data.get('n_add', '-')}`",
        f"- `Delete Count`: `{data.get('n_delete', '-')}`",
        f"- `Update Batch Size`: `{data.get('update_batch_size', '-')}`",
        f"- `Locality Samples`: `{data.get('locality_samples', '-')}`",
        f"- `Seed`: `{data.get('seed', '-')}`",
        f"- `Phase3 Weights`: `{_ckpt_label(data.get('phase3_weights'))}`",
        f"- `File`: `{_rel(path)}`",
        "",
    ]
    lines.extend(
        _table(
            ["指标", "数值"],
            [
                ["Add QA Acc Before", _pct(metrics.get("add_qa_acc_before"))],
                ["Add QA Acc After", _pct(metrics.get("add_qa_acc_after"))],
                ["Add Retrieval Top1 Before", _pct(metrics.get("add_retrieval_top1_before"))],
                ["Add Retrieval Top1 After", _pct(metrics.get("add_retrieval_top1_after"))],
                ["Delete QA Acc Before", _pct(metrics.get("delete_qa_acc_before"))],
                ["Delete QA Acc After", _pct(metrics.get("delete_qa_acc_after"))],
                ["Delete Retrieval Top1 Before", _pct(metrics.get("delete_retrieval_top1_before"))],
                ["Delete Retrieval Top1 After", _pct(metrics.get("delete_retrieval_top1_after"))],
                ["Old QA Retention", _pct(metrics.get("old_qa_retention_after_updates"))],
                ["Old Retrieval Retention", _pct(metrics.get("old_retrieval_retention_after_updates"))],
                ["Mean Add Latency (ms)", _num(metrics.get("mean_add_latency_ms"), 2)],
                ["Mean Delete Latency (ms)", _num(metrics.get("mean_delete_latency_ms"), 2)],
                ["Final Tombstone Ratio", _pct(metrics.get("tombstone_ratio_final"))],
            ],
        )
    )
    stage_rows: list[list[str]] = []
    if add_stage:
        first = add_stage[0]
        last = add_stage[-1]
        stage_rows.append([
            "Add",
            str(first.get("added_so_far", "-")),
            _pct(first.get("add_qa_acc")),
            _pct(first.get("add_retrieval_top1")),
            _pct(first.get("old_qa_acc")),
            _num(first.get("latency_ms"), 2),
        ])
        if last is not first:
            stage_rows.append([
                "Add Final",
                str(last.get("added_so_far", "-")),
                _pct(last.get("add_qa_acc")),
                _pct(last.get("add_retrieval_top1")),
                _pct(last.get("old_qa_acc")),
                _num(last.get("latency_ms"), 2),
            ])
    if delete_stage:
        first = delete_stage[0]
        last = delete_stage[-1]
        stage_rows.append([
            "Delete",
            str(first.get("deleted_so_far", "-")),
            _pct(first.get("delete_qa_acc")),
            _pct(first.get("delete_retrieval_top1")),
            _pct(first.get("old_qa_acc")),
            _num(first.get("latency_ms"), 2),
        ])
        if last is not first:
            stage_rows.append([
                "Delete Final",
                str(last.get("deleted_so_far", "-")),
                _pct(last.get("delete_qa_acc")),
                _pct(last.get("delete_retrieval_top1")),
                _pct(last.get("old_qa_acc")),
                _num(last.get("latency_ms"), 2),
            ])
    if stage_rows:
        lines.extend(["", "**Stage Summary**", ""])
        lines.extend(
            _table(
                ["Stage", "Updated", "Edit Acc", "Edit Top1", "Old QA Acc", "Latency (ms)"],
                stage_rows,
            )
        )
    return Section("E8", lines)


def _detect_experiment(path: Path, data: dict[str, Any]) -> str | None:
    parent = path.parent.name.lower()
    name = path.name.lower()
    exp_field = str(data.get("experiment", "")).lower()
    for token in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"):
        if token == parent or name.startswith(f"{token}_") or exp_field.startswith(token):
            return token.upper()
    return None


def _make_section(path: Path) -> Section | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    exp = _detect_experiment(path, data)
    if exp == "E1":
        return _summarize_e1(path, data)
    if exp == "E2":
        return _summarize_e2(path, data)
    if exp == "E3":
        return _summarize_e3(path, data)
    if exp == "E4":
        return _summarize_e4(path, data)
    if exp == "E5":
        return _summarize_e5(path, data)
    if exp == "E6":
        return _summarize_e6(path, data)
    if exp == "E7":
        if "summary" in data and "training_free_weights" in data:
            return _summarize_e7(path, data)
        if all(isinstance(data.get(ds), dict) and "results" in data.get(ds, {}) for ds in ("medqa", "arc", "mmlu")):
            return _summarize_e7_retrieval(path, data)
    if exp == "E8":
        kind = str(data.get("experiment", "")).lower()
        if kind == "e8a":
            return _summarize_e8a(path, data)
        if kind == "e8b":
            return _summarize_e8b(path, data)
        if kind == "e8c":
            return _summarize_e8c(path, data)
        if kind == "e8d_a":
            return _summarize_e8d_a(path, data)
        if kind == "e8d_b":
            return _summarize_e8d_b(path, data)
    return None


def build_summary() -> str:
    sections_by_exp: dict[str, list[Section]] = {f"E{i}": [] for i in range(1, 9)}
    all_paths = _scan_results()

    e1_paths = [path for path in all_paths if path.parent.name.lower() == "e1" or path.name.lower().startswith("e1_")]
    e1_section = _summarize_e1_group(e1_paths)
    if e1_section is not None:
        sections_by_exp["E1"].append(e1_section)

    e3_paths = [path for path in all_paths if path.parent.name.lower() == "e3" or path.name.lower().startswith("e3_")]
    e3_multik_section = _summarize_e3_multik(e3_paths)
    if e3_multik_section is not None:
        sections_by_exp["E3"].append(e3_multik_section)

    for path in all_paths:
        section = _make_section(path)
        if section is not None:
            if section.title == "E1":
                continue
            sections_by_exp[section.title].append(section)

    lines = [
        "# Results Summary",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "This file is auto-generated from `results/`. Keep `result.md` for hand-written conclusions.",
        "",
        *_model_overview(),
    ]

    for exp in (f"E{i}" for i in range(1, 9)):
        sections = sections_by_exp[exp]
        lines.append(f"## {exp}")
        lines.append("")
        if not sections:
            lines.append("No results found.")
            lines.append("")
            continue
        for index, section in enumerate(sections, start=1):
            if len(sections) > 1:
                lines.append(f"### {exp} Result {index}")
                lines.append("")
            lines.extend(section.lines)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_summary(), encoding="utf-8")
    print(f"[collect_results] wrote {output}")


if __name__ == "__main__":
    main()
