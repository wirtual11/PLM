"""
Training module for causal language model fine-tuning.

Uses Hugging Face Transformers Trainer with a GPT-2 architecture
trained from scratch on user-supplied text chunks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyper-parameters and paths needed to launch a training run."""

    # Model architecture
    model_name: str = "gpt2"
    tokenizer_name: str = "bert-base-uncased"
    vocab_size: int = 30_522  # bert-base-uncased default
    n_positions: int = 128  # max sequence length (>= max_tokens used in chunking)
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4

    # Training hyper-parameters
    epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    gradient_accumulation_steps: int = 1

    # Data
    eval_split: float = 0.1
    mlm_probability: float = 0.0  # 0 → causal LM (no masking)
    seed: int = 42
    max_seq_length: int = 128

    # I/O
    output_dir: str = "./output"
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2

    # Performance
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TextChunkDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A simple map-style dataset that tokenizes pre-chunked text
    and returns *input_ids*, *attention_mask*, and *labels* tensors.
    """

    def __init__(
        self,
        chunks: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        super().__init__()
        self._chunks: List[str] = chunks
        self._tokenizer: PreTrainedTokenizerBase = tokenizer
        self._max_length: int = max_length

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding: BatchEncoding = self._tokenizer(
            self._chunks[idx],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids: torch.Tensor = cast(torch.Tensor, encoding["input_ids"]).squeeze(0)
        attention_mask: torch.Tensor = cast(torch.Tensor, encoding["attention_mask"]).squeeze(0)

        # For causal LM the labels are identical to input_ids;
        # padding positions are set to -100 so they are ignored by the loss.
        labels: torch.Tensor = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training class
# ---------------------------------------------------------------------------


class Training:
    """
    End-to-end wrapper for training a small causal language model
    from a list of text chunks produced by the tokenizer pipeline.

    Usage
    -----
    >>> trainer = Training(all_chunks, config=TrainingConfig())
    >>> trainer.train()
    >>> trainer.save()
    """

    def __init__(
        self,
        chunks: List[str],
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self._config: TrainingConfig = config or TrainingConfig()
        self._chunks: List[str] = chunks

        # Resolve device
        self._device: torch.device = self._resolve_device()

        # Tokenizer
        self._tokenizer: PreTrainedTokenizerBase = self._build_tokenizer()

        # Datasets
        self._train_dataset: TextChunkDataset
        self._eval_dataset: TextChunkDataset
        self._train_dataset, self._eval_dataset = self._split_datasets()

        # Model (small GPT-2 initialised from scratch)
        self._model: GPT2LMHeadModel = self._build_model()

        # HF Trainer
        self._trainer: Trainer = self._build_trainer()

    # ----- public API -------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop and return training metrics."""
        result = self._trainer.train()  # type: ignore[no-untyped-call]
        raw_metrics: object = getattr(result, "metrics", {})  # type: ignore[no-untyped-usage]
        metrics: Dict[str, float] = cast(Dict[str, float], raw_metrics)
        self._trainer.log_metrics("train", metrics)  # type: ignore[no-untyped-call]
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the held-out split and return metrics incl. perplexity."""
        metrics: Dict[str, float] = cast(
            Dict[str, float],
            self._trainer.evaluate(),  # type: ignore[no-untyped-call]
        )
        try:
            eval_loss: float = metrics["eval_loss"]
            metrics["perplexity"] = math.exp(eval_loss)
        except (KeyError, OverflowError):
            metrics["perplexity"] = float("inf")
        self._trainer.log_metrics("eval", metrics)  # type: ignore[no-untyped-call]
        return metrics

    def save(self, path: Optional[str] = None) -> Path:
        """Persist model + tokenizer to *path* (defaults to output_dir/final)."""
        save_dir: Path = Path(path or os.path.join(self._config.output_dir, "final"))
        save_dir.mkdir(parents=True, exist_ok=True)
        self._trainer.save_model(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))  # type: ignore[no-untyped-call]
        return save_dir

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self._config.tokenizer_name,
                use_fast=True,
            ),
        )
        # GPT-2 style models need a pad token
        if tokenizer.pad_token is None:  # type: ignore[has-type]
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore[has-type]
        return tokenizer

    def _split_datasets(
        self,
    ) -> tuple[TextChunkDataset, TextChunkDataset]:
        """Split chunks into training and evaluation datasets."""
        total: int = len(self._chunks)
        eval_size: int = max(1, int(total * self._config.eval_split))
        train_size: int = total - eval_size

        generator: torch.Generator = torch.Generator().manual_seed(self._config.seed)
        indices: List[int] = cast(
            List[int],
            torch.randperm(total, generator=generator).tolist(),  # type: ignore[no-untyped-call]
        )

        train_chunks: List[str] = [self._chunks[i] for i in indices[:train_size]]
        eval_chunks: List[str] = [self._chunks[i] for i in indices[train_size:]]

        train_ds: TextChunkDataset = TextChunkDataset(
            train_chunks, self._tokenizer, self._config.max_seq_length,
        )
        eval_ds: TextChunkDataset = TextChunkDataset(
            eval_chunks, self._tokenizer, self._config.max_seq_length,
        )
        return train_ds, eval_ds

    def _build_model(self) -> GPT2LMHeadModel:
        """Create a small GPT-2 model initialised from scratch."""
        bos_id: int = int(self._tokenizer.bos_token_id or 0)  # type: ignore[arg-type]
        eos_id: int = int(self._tokenizer.eos_token_id or 0)  # type: ignore[arg-type]
        pad_id: int = int(self._tokenizer.pad_token_id or 0)  # type: ignore[arg-type]

        gpt2_config: GPT2Config = GPT2Config(
            vocab_size=self._config.vocab_size,
            n_positions=self._config.n_positions,
            n_embd=self._config.n_embd,
            n_layer=self._config.n_layer,
            n_head=self._config.n_head,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        model: GPT2LMHeadModel = GPT2LMHeadModel(gpt2_config)
        _ = model.to(self._device)  # type: ignore[no-untyped-call]
        return model

    def _build_trainer(self) -> Trainer:
        """Wire together HF Trainer with all components."""
        training_args: TrainingArguments = TrainingArguments(
            output_dir=self._config.output_dir,
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=self._config.per_device_train_batch_size,
            per_device_eval_batch_size=self._config.per_device_eval_batch_size,
            learning_rate=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
            warmup_ratio=self._config.warmup_ratio,
            max_grad_norm=self._config.max_grad_norm,
            fp16=self._config.fp16,
            bf16=self._config.bf16,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            eval_strategy=self._config.eval_strategy,
            save_strategy=self._config.save_strategy,
            save_total_limit=self._config.save_total_limit,
            logging_steps=self._config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self._config.seed,
            dataloader_num_workers=self._config.dataloader_num_workers,
            dataloader_pin_memory=self._config.dataloader_pin_memory,
            report_to="none",
        )

        data_collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,  # causal LM
        )

        trainer: Trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=data_collator,
        )
        return trainer
