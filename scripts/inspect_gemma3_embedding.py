"""
scripts/inspect_gemma3_embedding.py — 检查 Gemma3-1B 模型的 embedding 层结构

快速验证 Gemma3-1B 的内部层命名、embedding 结构，
评估与当前项目 Qwen3 wrapper 的兼容性。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/gemma-3-1b-it"  # 非 gated 镜像


def main() -> None:
    print(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    print("\n===== 顶层结构 =====")
    print(f"模型类: {type(model).__name__}")
    print(f"model.model 类型: {type(model.model).__name__}")

    # 顶层子模块
    for name, child in model.model.named_children():
        print(f"  model.model.{name}: {type(child).__name__}")

    # Embedding 层详情
    print("\n===== Embedding 层 =====")
    embed = model.model.embed_tokens
    print(f"类型: {type(embed).__name__}")
    print(f"权重 shape: {embed.weight.shape}")
    print(f"vocab_size: {embed.weight.shape[0]}")
    print(f"embedding_dim: {embed.weight.shape[1]}")
    print(f"dtype: {embed.weight.dtype}")

    # Transformer 层
    print("\n===== Transformer 层 =====")
    layers = model.model.layers
    print(f"总层数: {len(layers)}")

    # 第一层结构
    layer0 = layers[0]
    print(f"\n--- Layer 0 结构 ---")
    for name, child in layer0.named_children():
        print(f"  {name}: {type(child).__name__}")

    # Attention 子结构
    attn = layer0.self_attn
    print(f"\n--- Layer 0 Self-Attention ---")
    print(f"类型: {type(attn).__name__}")
    for name, param in attn.named_parameters():
        print(f"  {name}: {param.shape}")

    # MLP 子结构
    mlp = layer0.mlp
    print(f"\n--- Layer 0 MLP ---")
    for name, param in mlp.named_parameters():
        print(f"  {name}: {param.shape}")

    # Norm 层
    print(f"\n--- Layer 0 Input Norm ---")
    print(f"  {layer0.input_layernorm}")
    print(f"  weight shape: {layer0.input_layernorm.weight.shape}")

    # Final norm
    print(f"\n--- Final Norm ---")
    print(f"  类型: {type(model.model.norm).__name__}")
    print(f"  weight shape: {model.model.norm.weight.shape}")

    # rotary_emb
    print(f"\n--- Rotary Embedding ---")
    if hasattr(model.model, "rotary_emb"):
        print(f"  存在: model.model.rotary_emb")
        print(f"  类型: {type(model.model.rotary_emb).__name__}")
    else:
        print("  model.model.rotary_emb 不存在")
        # Gemma3 可能在各层内部有自己的 rotary_emb
        if hasattr(layer0.self_attn, "rotary_emb"):
            print(f"  在 self_attn 内部找到: {type(layer0.self_attn.rotary_emb).__name__}")

    # LM Head
    print(f"\n--- LM Head ---")
    print(f"  类型: {type(model.lm_head).__name__}")
    for name, param in model.lm_head.named_parameters():
        print(f"  {name}: {param.shape}")

    # 快速前向测试
    print("\n===== 前向测试 =====")
    input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
    print(f"输入: 'Hello, world!' → token IDs: {input_ids.tolist()}")

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    print(f"输出 logits shape: {outputs.logits.shape}")
    print(f"hidden states 层数: {len(outputs.hidden_states)}")
    print(f"  hidden_states[0] shape (embedding 输出): {outputs.hidden_states[0].shape}")
    last_hs = outputs.hidden_states[-1]
    print(f"  hidden_states[-1] shape (最后一层): {last_hs.shape}")

    # 对比 Qwen3 的关键接口
    print("\n===== 与 Qwen3 wrapper 兼容性检查 =====")
    checks = {
        "model.model.embed_tokens": hasattr(model.model, "embed_tokens"),
        "model.model.layers (ModuleList)": isinstance(model.model.layers, torch.nn.ModuleList),
        "model.model.norm": hasattr(model.model, "norm"),
        "model.model.rotary_emb": hasattr(model.model, "rotary_emb"),
        "layer.forward 接受 position_embeddings": False,  # 待确认
    }

    # 检查 layer forward 签名
    import inspect
    sig = inspect.signature(layer0.forward)
    params = list(sig.parameters.keys())
    checks["layer.forward 接受 position_embeddings"] = "position_embeddings" in params
    print(f"  Layer forward 参数: {params}")

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")


if __name__ == "__main__":
    main()
