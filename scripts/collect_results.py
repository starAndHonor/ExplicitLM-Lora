from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "results_summary.md"


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
        _table(
            ["File", "Weights", "Correct", "Counterfactual", "No Knowledge", "KS", "Total"],
            rows,
        ),
    )


def _summarize_e2(path: Path, data: dict[str, Any]) -> Section:
    rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = data.get(ds, {})
        rows.append([
            _rel(path),
            ds.upper(),
            _ckpt_label(data.get("weights")),
            _pct(ds_data.get("baseline", {}).get("acc")),
            _pct(ds_data.get("fusion_knowledge", {}).get("acc")),
            _pct(ds_data.get("fusion_empty", {}).get("acc")),
            _pct(ds_data.get("delta_acc")),
            _pct(ds_data.get("delta_acc_empty")),
        ])
    return Section(
        "E2",
        _table(
            ["File", "Dataset", "Weights", "Baseline", "Fusion+K", "Fusion+Empty", "Δacc", "Δempty"],
            rows,
        ),
    )


def _summarize_e3(path: Path, data: dict[str, Any]) -> Section:
    rows: list[list[str]] = []
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = data.get(ds, {})
        summary = data.get("summary", {}).get(ds, {})
        rows.append([
            _rel(path),
            ds.upper(),
            _ckpt_label(data.get("phase1_weights")),
            _ckpt_label(data.get("phase2_weights")),
            _pct(ds_data.get("G0_baseline", {}).get("acc")),
            _pct(ds_data.get("G1_rag_compressed", {}).get("acc")),
            _pct(ds_data.get("G2_fusion_phase1", {}).get("acc")),
            _pct(ds_data.get("G3_fusion_phase2", {}).get("acc")),
            _pct(ds_data.get("G4_rag_original", {}).get("acc")),
            _pct(summary.get("G3_vs_G1")),
        ])
    return Section(
        "E3",
        _table(
            ["File", "Dataset", "Phase1", "Phase2", "G0", "G1", "G2", "G3", "G4", "G3-G1"],
            rows,
        ),
    )


def _summarize_e4(path: Path, data: dict[str, Any]) -> Section:
    rows: list[list[str]] = []
    ablation = data.get("ablation", {})
    for ds in ("medqa", "arc", "mmlu"):
        ds_data = ablation.get(ds, {})
        sft_effect = ds_data.get("sft_effect")
        if sft_effect is None and ds_data.get("sft_cost") is not None:
            sft_effect = -ds_data.get("sft_cost")
        rows.append([
            _rel(path),
            ds.upper(),
            _ckpt_label(data.get("phase1_weights")),
            _ckpt_label(data.get("phase2_weights")),
            _pct(ds_data.get("baseline_acc")),
            _pct(ds_data.get("phase1_acc")),
            _pct(ds_data.get("phase2_acc")),
            _pct(ds_data.get("phase1_delta")),
            _pct(ds_data.get("phase2_delta")),
            _pct(sft_effect),
        ])
    return Section(
        "E4",
        _table(
            ["File", "Dataset", "Phase1", "Phase2", "Baseline", "Phase1 Acc", "Phase2 Acc", "ΔP1", "ΔP2", "SFT Effect"],
            rows,
        ),
    )


def _summarize_e5(path: Path, data: dict[str, Any]) -> Section:
    rows: list[list[str]] = []
    e5a = data.get("e5a", {})
    e5b = data.get("e5b", {})
    for ds in ("medqa", "arc", "mmlu"):
        a = e5a.get(ds, {})
        b = e5b.get(ds, {})
        rows.append([
            _rel(path),
            ds.upper(),
            _ckpt_label(data.get("phase1_weights")),
            _ckpt_label(data.get("phase2_weights")),
            _pct(a.get("baseline", {}).get("acc")),
            _pct(a.get("rag_k64", {}).get("acc")),
            _pct(a.get("fusion_p2_k64", {}).get("acc")),
            _pct(a.get("rag_k256", {}).get("acc")),
            _pct(b.get("oracle_p2", {}).get("acc")),
            _pct(b.get("shuffled_p2", {}).get("acc")),
            _pct(b.get("empty_p2", {}).get("acc")),
        ])
    return Section(
        "E5",
        _table(
            ["File", "Dataset", "Phase1", "Phase2", "Baseline", "RAG@64", "FusionP2@64", "RAG@256", "OracleP2", "ShuffledP2", "EmptyP2"],
            rows,
        ),
    )


def _summarize_e6(path: Path, data: dict[str, Any]) -> Section:
    benches = data.get("benchmarks", {})
    acc = data.get("accuracy_data", {}).get("medqa", {})
    rows = []
    for key in ("baseline", "rag_compressed", "fusion", "rag_original"):
        b = benches.get(key, {})
        rows.append([
            _rel(path),
            b.get("label", key),
            _ckpt_label(data.get("phase2_weights")),
            _num(b.get("latency_ms")),
            _num(b.get("throughput")),
            _num(b.get("peak_memory_mb"), 1),
            _num(b.get("avg_input_len"), 1),
            str(b.get("context_tokens", "-")),
        ])
    lines = _table(
        ["File", "Method", "Phase2", "Latency(ms)", "Throughput", "Memory(MB)", "Avg Input", "Ctx Tokens"],
        rows,
    )
    if acc:
        lines.extend([
            "",
            * _table(
                ["MedQA Ref", "Baseline", "RAG@64", "FusionP2@64", "RAG-original", "Fusion Phase2"],
                [[
                    _rel(path),
                    _pct(acc.get("baseline")),
                    _pct(acc.get("rag_k64")),
                    _pct(acc.get("fusion_p2_k64")),
                    _pct(acc.get("rag_original")),
                    _pct(acc.get("fusion_phase2")),
                ]],
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
    for path in _scan_results():
        section = _make_section(path)
        if section is not None:
            sections_by_exp[section.title].append(section)

    lines = [
        "# Results Summary",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "This file is auto-generated from `results/`. Keep `result.md` for hand-written conclusions.",
        "",
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
