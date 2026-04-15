#!/usr/bin/env python
"""
两轮对话式知识更新 + 自由文本生成

第一轮：输入原始知识文本
    → LLMLingua 压缩 → overlay 更新 dense index

第二轮：输入问题（自由文本，非选择题）
    → dense 检索（仅第一个 token 前检索一次）
    → 预编码 knowledge vector，整个 generation 复用
    → 自回归生成回答，流式输出

用法：
    CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM \\
        python scripts/chat_overlay_answer.py \\
        --base-index checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \\
        --phase3-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e1.counterfactual_eval import KnowledgeCompressor
from experiments.e2.common import build_injection_model, get_model_path
from experiments.e8.common import build_fusion_encoder_and_tokenizer, encode_fusion_texts
from models.modified_model import ModifiedQwen
from retrieval.dense_index import DenseKnowledgeIndex


# ─────────────────────────────────────────────────────────────────────────────
# UI 工具
# ─────────────────────────────────────────────────────────────────────────────

def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width, flush=True)

def _banner(title: str) -> None:
    _hr("═")
    print(f"  {title}", flush=True)
    _hr("═")

def _section(title: str) -> None:
    print(f"\n▶ {title}", flush=True)
    _hr("─", 50)

def _read_multiline(prompt: str) -> str:
    """读取多行输入，单次空行结束。"""
    print(prompt, flush=True)
    print("  （输入完毕后按一次 Enter 确认）", flush=True)
    _hr("·", 40)
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 压缩工具
# ─────────────────────────────────────────────────────────────────────────────

def _compress(
    compressor: Optional[KnowledgeCompressor],
    tokenizer: AutoTokenizer,
    text: str,
    fusion_length: int,
    rate: float,
    backend: str,
) -> tuple[str, List[int]]:
    """返回 (compressed_text, fusion_token_ids[fusion_length])。"""
    if backend == "mock_tokenize":
        ids = tokenizer.encode(text, add_special_tokens=False)[:fusion_length]
    else:
        assert compressor is not None
        compressed = compressor.compress_text(text)
        if not compressed:
            compressed = text[:200]
        ids = tokenizer.encode(compressed, add_special_tokens=False)[:fusion_length]
        text = compressed

    if len(ids) < fusion_length:
        ids += [tokenizer.pad_token_id] * (fusion_length - len(ids))

    decoded = tokenizer.decode(
        [x for x in ids if x != tokenizer.pad_token_id], skip_special_tokens=True
    ).strip()
    return decoded, ids


# ─────────────────────────────────────────────────────────────────────────────
# 生成核心：knowledge 只编码一次，后续 token 复用
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_cached_knowledge(
    model: ModifiedQwen,
    tokenizer: AutoTokenizer,
    prompt_ids: List[int],
    knowledge_ids: torch.Tensor,   # [1, K_f]
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    """
    自回归生成，knowledge 仅在第一个 token 前编码一次，后续 token 直接复用。

    实现原理：
        - 首轮调用前：knowledge_encoder(knowledge_ids) → k_encoded [1, K_f, D]
        - 每步生成前：直接写入 model._current_knowledge / _current_mask
          跳过 ModifiedQwen.forward() 中的重复编码
        - 调用 model.base_model() 触发已注册的 hook，完成注入
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # ── Step 0: 预编码 knowledge（只做一次）────────────────────────────
    k_mask = (knowledge_ids != pad_id).long()               # [1, K_f]
    k_encoded = model.knowledge_encoder(knowledge_ids.to(device), k_mask.to(device))  # [1, K_f, D]
    k_mask = k_mask.to(device)

    # ── 生成循环 ────────────────────────────────────────────────────────
    current_ids = list(prompt_ids)
    generated: List[int] = []

    for step in range(max_new_tokens):
        input_t = torch.tensor([current_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_t)

        # 直接注入预编码向量（跳过 knowledge_encoder 重复调用）
        model._current_knowledge = k_encoded
        model._current_mask = k_mask
        try:
            output = model.base_model(
                input_ids=input_t,
                attention_mask=attn_mask,
                use_cache=False,
            )
        finally:
            model._current_knowledge = None
            model._current_mask = None

        logits = output.logits[0, -1, :]          # [V]

        # 采样
        if temperature == 1.0 and top_p == 1.0:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_id = int(sorted_idx[torch.multinomial(sorted_probs, 1)].item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())

        if next_id == eos_id:
            break

        generated.append(next_id)
        current_ids.append(next_id)

        # 流式输出：逐 token 打印
        token_str = tokenizer.decode([next_id], skip_special_tokens=True)
        print(token_str, end="", flush=True)

    print(flush=True)  # 换行
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="两轮对话：第一轮喂入原始知识，第二轮喂入问题，生成自由文本回答"
    )
    p.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    p.add_argument("--base-index", required=True, help="基础 dense index 文件路径（.pt）")
    p.add_argument("--phase3-ckpt", required=True, help="Phase3 checkpoint 目录")
    p.add_argument(
        "--compression-backend", choices=["llmlingua", "mock_tokenize"], default="llmlingua",
    )
    p.add_argument("--compression-rate", type=float, default=0.25)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--override", nargs="?", action="append")
    return p.parse_args()


def _parse_overrides(overrides) -> Dict[str, Any]:
    if not overrides:
        return {}
    result: Dict[str, Any] = {}
    for item in (overrides or []):
        items = item if isinstance(item, list) else [item]
        for s in items:
            if "=" in s:
                k, v = s.split("=", 1)
                result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    device = torch.device(args.device)
    fusion_length = cfg.model.fusion_length
    anchor_length = cfg.model.fusion_length   # 单视图：统一使用 fusion_length

    _banner("两轮对话式知识更新 + 融合生成")
    print(f"  base_index   : {args.base_index}")
    print(f"  phase3_ckpt  : {args.phase3_ckpt}")
    print(f"  compression  : {args.compression_backend} (rate={args.compression_rate})")
    print(f"  generation   : max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}")

    # ── 加载组件 ──────────────────────────────────────────────────────────
    _section("初始化模型组件")

    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  tokenizer     ✓")

    encoder, enc_tok = build_fusion_encoder_and_tokenizer(cfg, device=device)
    print(f"  encoder       ✓  (retrieval_depth={cfg.model.retrieval_encoder_depth}, fusion_depth={cfg.model.fusion_encoder_depth})")

    dense_index = DenseKnowledgeIndex.load(args.base_index)
    print(f"  dense index   ✓  (total={len(dense_index)}, active={dense_index.num_active})")

    model, _ = build_injection_model(
        cfg, fusion_ckpt=args.phase3_ckpt, device=str(device), log_prefix="Chat"
    )
    model.eval()
    print(f"  injection model ✓")

    compressor: Optional[KnowledgeCompressor] = None
    if args.compression_backend == "llmlingua":
        gpu_id = device.index if device.type == "cuda" else None
        compressor = KnowledgeCompressor(
            model_name=cfg.paths.llmlingua_model_dir,
            compression_rate=args.compression_rate,
            gpu_id=gpu_id,
        )
        print(f"  llmlingua     ✓  ({cfg.paths.llmlingua_model_dir})")

    print("\n  所有组件加载完毕 ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # 第一轮：输入原始知识（不含答案）
    # ══════════════════════════════════════════════════════════════════════════
    _banner("【第一轮】输入原始知识（不含答案）")
    print("  粘贴或输入原始知识文本，以空行结束。")
    print("  ⚠  请勿包含正确答案，只输入背景事实/场景描述。\n")

    raw_knowledge = _read_multiline("原始知识：")
    if not raw_knowledge.strip():
        print("[错误] 知识文本不能为空。")
        sys.exit(1)

    # 压缩
    _section("LLMLingua 压缩中...")
    compressed_text, fusion_ids = _compress(
        compressor=compressor,
        tokenizer=tokenizer,
        text=raw_knowledge,
        fusion_length=fusion_length,
        rate=args.compression_rate,
        backend=args.compression_backend,
    )
    valid_tok = sum(1 for x in fusion_ids if x != tokenizer.pad_token_id)
    print(f"  原始     : {len(raw_knowledge.split())} words")
    print(f"  压缩结果 : {compressed_text}")
    print(f"  有效 token : {valid_tok} / {fusion_length}")

    # Overlay
    _section("Overlay 更新知识库")
    knowledge_key = raw_knowledge[:200].strip()
    fusion_ids_t = torch.tensor([fusion_ids], dtype=torch.long)
    # 单视图：embedding 来自 compressed_text（与 fusion_ids 同源）
    embedding = encode_fusion_texts(cfg=cfg, encoder=encoder, tokenizer=enc_tok,
                                    texts=[compressed_text], device=device)  # [1, D]

    rng = random.Random(args.seed)
    replaced_key = dense_index.keys[rng.randrange(len(dense_index))]
    dense_index.delete_by_keys([replaced_key])
    dense_index.add_entries(
        embeddings=embedding, fusion_ids=fusion_ids_t,
        keys=[knowledge_key], texts=[raw_knowledge], replace_existing=True,
    )
    dense_index.compact()
    print(f"  替换条目 : {replaced_key[:70]}...")
    print(f"  新写入   : {knowledge_key[:70]}...")
    print(f"  index 规模 : {len(dense_index)} (active={dense_index.num_active})")
    print("\n  知识库更新完毕 ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # 第二轮：输入问题，生成回答
    # ══════════════════════════════════════════════════════════════════════════
    _banner("【第二轮】输入问题")

    question = input("问题：").strip()
    if not question:
        print("[错误] 问题不能为空。")
        sys.exit(1)

    # 检索 query = 问题 + 第一轮压缩知识（不带答案，保证命中新写入条目）
    # compressed_text 来自不含答案的原始知识，避免答案信息泄露到检索路径
    retrieval_query = f"{question} {compressed_text}".strip()
    print(f"  检索 query（问题 + 压缩知识，不含答案）: {retrieval_query[:120]}...")

    # 检索（仅此一次）
    _section("Dense 检索（第一个 token 前，仅检索一次）")
    enc_q = enc_tok(
        [retrieval_query],
        max_length=anchor_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = encoder.encode_mean(
            enc_q["input_ids"].to(device),
            enc_q["attention_mask"].long().to(device),
        )
    search_out = dense_index.search(q_emb.detach().cpu(), top_k=1)

    idx = int(search_out.indices[0, 0].item())
    score = float(search_out.scores[0, 0].item())
    retrieved_key = dense_index.keys[idx] if idx >= 0 else "N/A"
    retrieved_fusion = search_out.fusion_ids[0, 0]  # [K_f]
    hit_marker = " ◀ 命中新知识 ✓" if retrieved_key == knowledge_key else ""

    print(f"  检索得分 : {score:.4f}")
    print(f"  命中 key : {retrieved_key[:70]}...{hit_marker}")
    decoded_knowledge = tokenizer.decode(
        [x for x in retrieved_fusion.tolist() if x != tokenizer.pad_token_id],
        skip_special_tokens=True,
    ).strip()
    print(f"  注入知识 : {decoded_knowledge[:100]}...")
    print(f"\n  knowledge 预编码完毕，后续所有 token 直接复用 ✓")

    # 构建 prompt（简洁格式，与 Phase3 训练格式对齐）
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    knowledge_ids_for_gen = retrieved_fusion.unsqueeze(0)  # [1, K_f]

    # 生成
    _section("生成回答（流式输出）")
    print(f"\nAnswer: ", end="", flush=True)

    answer = generate_with_cached_knowledge(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        knowledge_ids=knowledge_ids_for_gen,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    _hr("═")
    print(f"  生成完成，共 {len(answer.split())} words")
    _hr("═")


if __name__ == "__main__":
    main()
