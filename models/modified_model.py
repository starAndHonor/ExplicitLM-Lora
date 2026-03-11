"""
models/modified_model.py — ModifiedQwen：Hook 注入式知识融合模型

通过 PyTorch register_forward_hook 在 Qwen3 指定层（默认 [6,12,18,24]）
非侵入式注入知识向量，实现基础模型完全冻结、仅注入模块参与训练的融合架构。

核心设计：
  - Hook 机制：无需修改 Qwen3 源码，直接在指定层输出后调用 AttentionInjection
  - 知识一次编码：forward 开始时调用 KnowledgeEncoder 一次，所有注入层复用
  - 零初始化安全：AttentionInjection 初始 out_proj=0 → 训练初期等价于原始模型
  - 退化模式：knowledge_ids=None 时 hook 跳过注入，完全等价于原始 Qwen3

前向数据流：
    input_ids [B, L] + knowledge_ids [B, K_f]
         │                     │
         │        knowledge_encoder → [B, K_f, D]（一次编码）
         │                     │
         │        Qwen3 Layer 0-5（无注入）
         │                     │
         └──────── Hook ───────┤ Layer 6  → AttentionInjection(hidden, knowledge)
                               │
                        Layer 7-11
                               │
                          Hook ─┤ Layer 12 → AttentionInjection
                              ...（Layer 18, 24 同理）
                               │
                        lm_head → logits [B, L, V]

依赖：transformers, models/injection_modules.py, models/qwen_wrapper.py
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.injection_modules import BaseInjection
from models.qwen_wrapper import KnowledgeEncoder

logger = logging.getLogger(__name__)


class ModifiedQwen(nn.Module):
    """
    Qwen3 + Hook 注入，在 injection_layers 位置调用 AttentionInjection。

    基础模型完全冻结，仅 injection_modules（与可选的 KnowledgeEncoder 前 N 层）参与训练。

    参数：
        base_model: 已冻结的 Qwen3 AutoModelForCausalLM 实例（由 load_base_model 返回）
        knowledge_encoder: KnowledgeEncoder 实例，将 knowledge_ids 编码为 [B, K_f, D]
        injection_modules: nn.ModuleList，长度须与 injection_layers 一致，每层一个 BaseInjection
        injection_layers: 注入点层索引列表，默认 [6, 12, 18, 24]（Qwen3-0.6B 共 28 层）
        pad_token_id: padding token ID，用于从 knowledge_ids 推断 knowledge_mask（默认 0）
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        knowledge_encoder: KnowledgeEncoder,
        injection_modules: nn.ModuleList,
        injection_layers: List[int],
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        assert len(injection_modules) == len(injection_layers), (
            f"injection_modules 长度 {len(injection_modules)} 须等于 "
            f"injection_layers 长度 {len(injection_layers)}"
        )

        # Phase 1: 存储子模块（注册为 nn.Module 属性，确保 .parameters()/.to() 正常）
        self.base_model = base_model
        self.knowledge_encoder = knowledge_encoder
        self.injection_modules = injection_modules
        self.injection_layers = injection_layers
        self.pad_token_id = pad_token_id

        # Phase 2: 确保基础模型全量冻结
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Phase 3: 注册 post-hook 到指定层
        # Qwen3DecoderLayer.forward 在 use_cache=False 时直接返回 Tensor（非元组）
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        for i, layer_idx in enumerate(injection_layers):
            handle = self.base_model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(handle)

        # Phase 4: 运行时知识上下文（forward 前设置，forward 后清空）
        self._current_knowledge: Optional[Tensor] = None
        self._current_mask: Optional[Tensor] = None

        logger.info(
            "ModifiedQwen 初始化完毕 (injection_layers=%s, injection_method=%s)",
            injection_layers,
            type(injection_modules[0]).__name__ if injection_modules else "None",
        )

    def _make_hook(self, module_idx: int) -> Callable:
        """
        生成第 module_idx 个注入点的 post-hook 闭包。

        hook 在对应 Qwen3DecoderLayer 的 forward 结束后自动调用，
        将层输出的隐藏状态注入知识向量。

        参数：
            module_idx: 注入点索引（对应 injection_modules[module_idx]）

        返回：
            hook(module, input, output) → Optional[Tensor]
                若 _current_knowledge 为 None（退化模式），返回 None（不修改输出）
                否则调用 injection_modules[module_idx] 并返回注入后的隐藏状态
        """

        def hook(
            module: nn.Module,
            input: tuple,
            output: Tensor,
        ) -> Optional[Tensor]:
            # 退化模式：knowledge_ids=None 时跳过注入
            if self._current_knowledge is None:
                return None

            # 调用对应注入模块：hidden [B, L, D] + knowledge [B, K_f, D] → [B, L, D]
            # injection_modules 以 float32 运行（高精度梯度），输出需转回 output.dtype
            # 确保后续 Qwen3 层（bf16 权重）能接受正确 dtype 的 hidden states
            injected = self.injection_modules[module_idx](
                output,
                self._current_knowledge,
                self._current_mask,
            )
            return injected.to(output.dtype)

        return hook

    def forward(
        self,
        input_ids: Tensor,                    # [B, L] LongTensor
        knowledge_ids: Optional[Tensor],      # [B, K_f] LongTensor，None 表示退化模式
        attention_mask: Tensor,               # [B, L] LongTensor，1=有效 0=padding
        labels: Optional[Tensor] = None,      # [B, L] LongTensor，-100 为忽略位
    ) -> CausalLMOutputWithPast:
        """
        知识注入前向传播。

        参数：
            input_ids: [B, L] 主文本 token IDs
            knowledge_ids: [B, K_f] 压缩知识 token IDs（None 时退化为原始 Qwen3）
            attention_mask: [B, L] 主文本注意力掩码
            labels: [B, L] 目标 token IDs（训练时提供，-100 位置忽略损失）

        返回：
            CausalLMOutputWithPast：
                .logits: [B, L, V] 预测 logits
                .loss: 若提供 labels 则包含交叉熵损失（标量）

        实现步骤：
            Step 1: 知识编码（knowledge_ids 非 None 时）
                knowledge_mask = (knowledge_ids != pad_token_id)  → [B, K_f]
                _current_knowledge = knowledge_encoder(knowledge_ids, knowledge_mask)  → [B, K_f, D]
                _current_mask = knowledge_mask
            Step 2: 基础模型前向（Hook 自动在指定层注入）
                use_cache=False 确保 DecoderLayer 返回 Tensor（非元组），hook 可直接处理
            Step 3: 清空知识上下文并返回
        """
        try:
            # Step 1: 知识编码（一次，所有注入层复用）
            if knowledge_ids is not None:
                knowledge_mask = (knowledge_ids != self.pad_token_id).long()  # [B, K_f]
                self._current_knowledge = self.knowledge_encoder(
                    knowledge_ids, knowledge_mask
                )  # [B, K_f, D]
                self._current_mask = knowledge_mask
            else:
                # 退化模式：_current_knowledge=None，hook 自动跳过注入
                self._current_knowledge = None
                self._current_mask = None

            # Step 2: 基础模型前向，hook 自动触发注入
            # use_cache=False：确保 DecoderLayer 返回 Tensor（与 KnowledgeEncoder 一致）
            output: CausalLMOutputWithPast = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

        finally:
            # Step 3: 清空知识上下文，防止下次 forward 误用
            self._current_knowledge = None
            self._current_mask = None

        return output

    def remove_hooks(self) -> None:
        """
        移除所有已注册的 forward hook。

        用于测试 teardown 或需要恢复原始 Qwen3 行为时调用。
        调用后 ModifiedQwen 退化为透传原始 base_model 的薄包装。
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.info("已移除所有 forward hooks（共 %d 个）", len(self.injection_layers))
