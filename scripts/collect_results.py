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
    rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = data.get(ds, {})
        total = ds_data.get("baseline", {}).get("total", "-")
        if ds == "medqa":
            ds_label = f"MedQA（{total:,} 题）" if isinstance(total, int) else "MedQA"
        elif ds == "arc":
            ds_label = f"ARC-Challenge（{total:,} 题）" if isinstance(total, int) else "ARC-Challenge"
        else:
            ds_label = f"MMLU（{total:,} 题）" if isinstance(total, int) else "MMLU"
        rows.append([
            ds_label,
            _pct(ds_data.get("baseline", {}).get("acc")),
            _pct(ds_data.get("fusion_knowledge", {}).get("acc")),
            _pct(ds_data.get("fusion_empty", {}).get("acc")),
            ("+" if isinstance(ds_data.get("delta_acc"), (int, float)) and ds_data.get("delta_acc") >= 0 else "") + _pct(ds_data.get("delta_acc")),
        ])
    return Section(
        "E2",
        _config_summary(data)
        + _table(
            ["数据集", "Baseline", "Fusion+知识", "Fusion+空知识", "Δacc"],
            rows,
        ),
    )


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


def _detect_experiment(path: Path, data: dict[str, Any]) -> str | None:
    parent = path.parent.name.lower()
    name = path.name.lower()
    exp_field = str(data.get("experiment", "")).lower()
    for token in ("e1", "e2", "e3", "e4", "e5", "e6"):
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
    return None


def build_summary() -> str:
    sections_by_exp: dict[str, list[Section]] = {f"E{i}": [] for i in range(1, 7)}
    all_paths = _scan_results()

    e1_paths = [path for path in all_paths if path.parent.name.lower() == "e1" or path.name.lower().startswith("e1_")]
    e1_section = _summarize_e1_group(e1_paths)
    if e1_section is not None:
        sections_by_exp["E1"].append(e1_section)

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

    for exp in (f"E{i}" for i in range(1, 7)):
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
