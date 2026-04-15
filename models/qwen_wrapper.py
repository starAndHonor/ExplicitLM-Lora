"""
models/qwen_wrapper.py — Qwen3 知识编码器

封装 Qwen3 前 encoder_depth 层作为知识编码器，将来自 FusionBank 的
token IDs 上下文化为稠密知识表示 [B, K_f, D]，供 AttentionInjection 注入。

核心设计：
  - trainable 模式：显式 mask + 可训练独立 norm + 前 encoder_depth 层可联合训练
  - qwen3 模式：完全复用旧版 Qwen helper 语义，不训练 encoder
"""

from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_base_model(model_path: str, bf16: bool = True) -> AutoModelForCausalLM:
    """
    加载 Qwen3 基础模型并冻结所有参数。

    参数：
        model_path: 模型本地路径或 HuggingFace Hub 名称
        bf16: 是否使用 bfloat16 精度（True 节省显存，False 使用 float32）

    返回：
        完全冻结（requires_grad=False）的 AutoModelForCausalLM 实例

    示例：
        >>> model = load_base_model("Qwen3-0.6B", bf16=True)
    """
    dtype = torch.bfloat16 if bf16 else torch.float32
    logger.info("加载基础模型: %s (dtype=%s)", model_path, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # 冻结所有参数（基础模型在整个项目中保持冻结）
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    logger.info("基础模型加载完毕，共 %d 个参数已冻结", sum(1 for _ in model.parameters()))

    return model


class KnowledgeEncoder(nn.Module):
    """
    Qwen3 前 encoder_depth 层 + 独立 RMSNorm，将知识 token 序列上下文化。

    从基础模型引用（共享权重）前 N 层，初始全部冻结。
    Phase 2 开始时调用 unfreeze_layers() 解冻前 N 层与注入模块联合训练。
    独立创建的 self.norm（深拷贝）在整个训练期间均可训练。

    数据流：
        knowledge_ids [B, K_f] → embed_tokens → 前 N 层（双向注意力）→ norm → [B, K_f, D]

    参数：
        base_model: 已冻结的 Qwen3 AutoModelForCausalLM 实例（由 load_base_model 返回）
        encoder_depth: 使用 Qwen3 前多少层作为编码器（默认 6）
        hidden_dim: 模型隐藏维度（Qwen3-0.6B 为 1024）
        mode:
            - "trainable"：当前主线模式，显式 mask + 独立 norm + 可联合训练
            - "qwen3"：复用 Qwen encoder helper 语义且不训练
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        encoder_depth: int,
        hidden_dim: int,
        mode: str = "trainable",
    ) -> None:
        super().__init__()

        self.encoder_depth = encoder_depth
        self.hidden_dim = hidden_dim
        self.mode = mode.lower()
        if self.mode == "reference":
            self.mode = "qwen3"
        if self.mode not in {"trainable", "qwen3"}:
            raise ValueError(
                f"unsupported knowledge encoder mode: {mode} "
                "(expected 'trainable' or 'qwen3')"
            )

        # Phase 1: 引用共享组件（与 base_model 共享同一批权重对象）
        # embed_tokens: 词嵌入层，始终冻结
        self.embed_tokens = base_model.model.embed_tokens
        # layers: 前 encoder_depth 个 Transformer 层，初始冻结，Phase 2 解冻
        self.layers = nn.ModuleList(list(base_model.model.layers[:encoder_depth]))
        # rotary_emb: 旋转位置编码，模型级别，Qwen3 此版本 forward 要求外部传入 position_embeddings
        # （不可删除：Qwen3Attention.forward 直接解包 position_embeddings，不接受 None）
        self.rotary_emb = base_model.model.rotary_emb

        if self.mode == "trainable":
            # trainable: 深拷贝 Final RMSNorm，作为独立可训练参数（不共享 base_model 权重）
            self.norm: nn.Module = copy.deepcopy(base_model.model.norm)
            for p in self.norm.parameters():
                p.requires_grad = True
        else:
            # qwen3: 直接复用 base model 的 final norm，不训练 encoder
            self.norm = base_model.model.norm

        # Phase 3: 初始化时冻结所有共享组件
        self._freeze_all()

        if self.encoder_depth <= 0:
            layer_usage = "embedding_only"
            retrieval_path = "embed_tokens -> final_norm -> mean_pool"
        else:
            last_layer = self.encoder_depth - 1
            layer_usage = f"layers[0:{self.encoder_depth}] (0-{last_layer})"
            retrieval_path = f"embed_tokens -> layers 0-{last_layer} -> final_norm -> mean_pool"

        logger.info(
            "KnowledgeEncoder 初始化完毕 (encoder_depth=%d, hidden_dim=%d, mode=%s, layer_usage=%s)",
            encoder_depth,
            hidden_dim,
            self.mode,
            layer_usage,
        )
        logger.info("KnowledgeEncoder 检索路径: %s", retrieval_path)

    def _freeze_all(self) -> None:
        """
        冻结所有共享组件参数（embed_tokens、layers、rotary_emb）。

        由 __init__ 内部调用，无需外部手动调用。
        trainable 模式下 self.norm 保持 requires_grad=True。
        qwen3 模式下 self.norm 同样保持冻结。
        """
        for p in self.embed_tokens.parameters():
            p.requires_grad = False

        for p in self.layers.parameters():
            p.requires_grad = False

        # rotary_emb 通常无 learnable 参数，保险起见仍冻结
        for p in self.rotary_emb.parameters():
            p.requires_grad = False

        if self.mode == "qwen3":
            for p in self.norm.parameters():
                p.requires_grad = False

    def unfreeze_layers(self) -> None:
        """
        解冻前 encoder_depth 层，供 Phase 2 与 AttentionInjection 联合训练。

        调用后效果：
            - self.layers 的所有参数 requires_grad = True
            - self.embed_tokens 仍然冻结（requires_grad=False）
            - self.norm 保持可训练（不变）
        """
        if self.mode == "qwen3":
            logger.info("KnowledgeEncoder 处于 qwen3 模式，跳过解冻 layers")
            return
        for p in self.layers.parameters():
            p.requires_grad = True
        logger.info("已解冻前 %d 层（联合训练模式已激活）", self.encoder_depth)

    @property
    def uses_qwen3_mode(self) -> bool:
        """是否处于 qwen3 兼容模式。"""
        return self.mode == "qwen3"

    @property
    def uses_reference_mode(self) -> bool:
        """兼容旧调用名；等价于 uses_qwen3_mode。"""
        return self.uses_qwen3_mode

    @property
    def device(self) -> torch.device:
        """
        返回编码器当前所在设备。

        供 AnchorBank.get_embeddings 中 .to(encoder.device) 调用。
        通过遍历所有参数找到第一个有参数的模块来确定设备。
        """
        return next(self.parameters()).device

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor,  # [B, K]
        dtype: torch.dtype,
    ) -> torch.Tensor:  # [B, 1, 1, K]
        """
        将 padding mask 转为 additive attention bias（双向注意力，无 causal mask）。

        知识编码使用双向注意力（Encoder 模式），所有有效 token 可互相 attend，
        但不 attend padding 位置。

        参数：
            attention_mask: [B, K] LongTensor，1=有效 token，0=padding
            dtype: 目标精度（与 hidden states 一致，避免精度不匹配）

        返回：
            [B, 1, 1, K] additive bias tensor
                有效位置 = 0.0（不遮蔽）
                padding 位置 = -inf（完全遮蔽）
        """
        # 扩展到 [B, 1, 1, K] 对 key 维度施加 mask
        extended = attention_mask[:, None, None, :].to(dtype=dtype)  # [B, 1, 1, K]

        # 0（padding）→ -inf，1（有效）→ 0.0
        extended = (1.0 - extended) * torch.finfo(dtype).min

        return extended

    def forward(
        self,
        knowledge_ids: torch.Tensor,   # [B, K_f] LongTensor
        attention_mask: torch.Tensor,  # [B, K_f] LongTensor，1=有效，0=pad
    ) -> torch.Tensor:                 # [B, K_f, D] FloatTensor
        """
        将知识 token IDs 编码为上下文化的稠密 token 级表示。

        参数：
            knowledge_ids: [B, K_f] LongTensor，来自 FusionBank 的压缩 token IDs
            attention_mask: [B, K_f] LongTensor，0 为 padding 位置

        返回：
            [B, K_f, D] FloatTensor，每个 token 位置的上下文化知识表示

        实现步骤：
            Phase 1: 词嵌入查找
            Phase 2: 构造位置 ID（顺序整数）
            Phase 3: 计算旋转位置编码（cos, sin），取第一层 rotary_emb 各层复用
            Phase 4:
                - trainable：构造 additive attention bias（双向，无 causal mask）
                - qwen3：不传显式 mask，完全复用旧版 Qwen helper 语义
            Phase 5: 逐层 Transformer 前向
            Phase 6: RMSNorm 归一化
        """
        b, k = knowledge_ids.shape

        grad_ctx = torch.no_grad() if self.uses_qwen3_mode else nullcontext()
        with grad_ctx:
            # Phase 1: 词嵌入查找  [B, K_f] → [B, K_f, D]
            h = self.embed_tokens(knowledge_ids)

            # Phase 2: 构造位置 ID  [B, K_f]
            position_ids = (
                torch.arange(k, device=knowledge_ids.device)
                .unsqueeze(0)
                .expand(b, -1)
            )

            # Phase 3: 计算旋转位置编码（cos, sin），供各层复用
            # Qwen3 此版本 Attention.forward 直接解包 position_embeddings，不接受 None
            cos, sin = self.rotary_emb(h, position_ids)

            if self.uses_qwen3_mode:
                # 完全复用 Reference helper 语义：不传显式 attention_mask/pad mask
                for layer in self.layers:
                    h = layer(h, position_embeddings=(cos, sin))
            else:
                # trainable: 构造 additive attention bias（双向注意力）  [B, 1, 1, K_f]
                attn_bias = self._build_attention_mask(attention_mask, dtype=h.dtype)

                for layer in self.layers:
                    h = layer(
                        h,
                        attention_mask=attn_bias,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=None,
                        position_embeddings=(cos, sin),
                    )  # [B, K_f, D]

            # Phase 6: RMSNorm 归一化  [B, K_f, D]
            h = self.norm(h)

        return h

    def encode_mean(
        self,
        knowledge_ids: torch.Tensor,   # [B, K] LongTensor
        attention_mask: torch.Tensor,  # [B, K] LongTensor，1=有效，0=pad
    ) -> torch.Tensor:                 # [B, D] FloatTensor
        """
        编码知识序列并做 masked mean pooling，返回句子级稠密表示。

        供 AnchorBank.get_embeddings 调用，用于聚类路由键更新。
        支持任意序列长度 K（K_f 或 K_a 均可）。

        参数：
            knowledge_ids: [B, K] LongTensor
            attention_mask: [B, K] LongTensor，0 为 padding

        返回：
            [B, D] FloatTensor，masked mean pooling 后的句子表示

        实现：
            Phase 1: 编码为 token 级表示 [B, K, D]
            Phase 2: 对有效 token 做 mean pooling（排除 padding 位置）
        """
        # Phase 1: 编码为 token 级表示  [B, K, D]
        h = self.forward(knowledge_ids, attention_mask)

        # Phase 2: Masked mean pooling（排除 padding 位置）
        mask_float = attention_mask.float().unsqueeze(-1)    # [B, K, 1]
        h_masked = h * mask_float                            # [B, K, D]
        token_count = mask_float.sum(dim=1).clamp(min=1.0)  # [B, 1]，避免全 pad 除零
        mean_h = h_masked.sum(dim=1) / token_count          # [B, D]

        return mean_h
