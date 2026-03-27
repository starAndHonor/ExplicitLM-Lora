"""MedQA 知识构建器。

为 MedQA 每道题预计算压缩知识（question + answer → LLMLingua-2 → knowledge_ids）。
生成的知识映射供 ExplicitQwenLM 评测时使用。

知识来源: question + correct_answer（模拟 RAG 检索到完美知识的场景）
Key 格式: sent1[:200]（与 lm-eval 的 doc_to_text 对齐）
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from logger_system import log_msg


class MedQAKnowledgeBuilder:
    """MedQA 知识构建器。

    流程:
    1. 加载 MedQA-USMLE-4-options-hf 数据集
    2. 对每道题: compress(question + correct_answer) → knowledge_ids[64]
    3. 保存为 JSONL（key → knowledge_ids 映射）
    """

    def __init__(
        self,
        tokenizer_path: str,
        compressor_model: str,
        knowledge_length: int = 64,
        gpu_id: int = 4,
    ):
        """初始化知识构建器。

        Args:
            tokenizer_path: tokenizer 路径（与训练一致，如 qwen3-0.6B）
            compressor_model: LLMLingua-2 模型路径
            knowledge_length: 知识 token 长度（与训练一致，默认 64）
            gpu_id: LLMLingua-2 使用的 GPU
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 延迟 import，避免 llmlingua 依赖链在非构建场景下阻塞
        from data_builder.compressor import KnowledgeCompressor

        self.compressor = KnowledgeCompressor(
            model_name=compressor_model,
            compression_rate=0.25,
            gpu_id=gpu_id,
        )
        self.knowledge_length = knowledge_length

        log_msg("INFO", f"知识构建器初始化完成 | GPU={gpu_id}, K={knowledge_length}")

    def build(
        self,
        dataset_name: str = "GBaker/MedQA-USMLE-4-options-hf",
        split: str = "test",
    ) -> Dict[str, List[int]]:
        """构建知识映射。

        从 HF cache 加载 MedQA-USMLE-4-options-hf 数据集（需提前缓存）。
        字段: sent1, ending0-3, label。

        Args:
            dataset_name: HuggingFace 数据集名称
            split: 数据集 split（默认 test）

        Returns:
            {key: knowledge_ids} 映射，key = sent1[:200]
        """
        ds = load_dataset(dataset_name, split=split)

        knowledge_map: Dict[str, List[int]] = {}
        total = len(ds)
        failed = 0

        log_msg("INFO", f"开始构建知识 | 总数: {total}, split: {split}")

        for i, row in enumerate(ds):
            # Phase 1: 拼接 question + correct_answer
            question = row["sent1"]
            label = row["label"]
            correct_answer = row[f"ending{label}"]
            source_text = f"{question} {correct_answer}"

            # Phase 2: LLMLingua-2 压缩
            compressed = self.compressor.compress_text(source_text)
            if compressed is None:
                failed += 1
                compressed = question[:100]  # fallback: 用问题前100字符

            # Phase 3: tokenize + pad/truncate 到 knowledge_length
            tokens = self.tokenizer.encode(compressed, add_special_tokens=False)
            tokens = tokens[: self.knowledge_length]
            if len(tokens) < self.knowledge_length:
                tokens = tokens + [self.tokenizer.pad_token_id] * (
                    self.knowledge_length - len(tokens)
                )

            # Key: sent1 前 200 字符
            key = question[:200].strip()
            knowledge_map[key] = tokens

            if (i + 1) % 100 == 0:
                log_msg("INFO", f"知识构建进度: {i+1}/{total}")

        log_msg(
            "INFO",
            f"知识构建完成 | 成功: {total - failed}, 失败: {failed}",
        )
        return knowledge_map

    @staticmethod
    def save(knowledge_map: Dict[str, List[int]], output_path: str) -> None:
        """保存知识映射为 JSONL。

        Args:
            knowledge_map: {key: knowledge_ids} 映射
            output_path: 输出文件路径
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            for key, ids in knowledge_map.items():
                f.write(json.dumps({"key": key, "knowledge_ids": ids}) + "\n")

        log_msg("INFO", f"知识映射保存: {output} ({len(knowledge_map)} 条)")

    @staticmethod
    def load(path: str) -> Dict[str, List[int]]:
        """加载知识映射。

        Args:
            path: JSONL 文件路径

        Returns:
            {key: knowledge_ids} 映射
        """
        knowledge_map: Dict[str, List[int]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                knowledge_map[entry["key"]] = entry["knowledge_ids"]

        log_msg("INFO", f"知识映射加载: {path} ({len(knowledge_map)} 条)")
        return knowledge_map
