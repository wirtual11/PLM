"""
Interactive console chat with a trained causal language model.

Loads a saved GPT-2 checkpoint and runs an interactive generate loop
so the user can prompt the model from the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ChatConfig:
    """Generation hyper-parameters for the interactive chat."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    do_sample: bool = True


class Chat:
    """
    Interactive console chat powered by a trained GPT-2 causal LM.

    Usage
    -----
    >>> chat = Chat(model_dir="./output/final")
    >>> chat.start()
    """

    QUIT_COMMANDS: set[str] = {"quit", "exit", "q"}

    def __init__(
        self,
        model_dir: str,
        config: Optional[ChatConfig] = None,
    ) -> None:
        self._model_dir: Path = Path(model_dir)
        self._config: ChatConfig = config or ChatConfig()
        self._device: torch.device = self._resolve_device()
        self._tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()
        self._model: GPT2LMHeadModel = self._load_model()

    # ----- public API -------------------------------------------------------

    def start(self) -> None:
        """Run the interactive read-eval-print loop in the console."""
        print("\n" + "=" * 60)
        print("  Chat with your model  (type 'quit' to exit)")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input: str = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in self.QUIT_COMMANDS:
                print("Goodbye!")
                break

            response: str = self.generate(user_input)
            print(f"Model: {response}\n")

    def generate(self, prompt: str) -> str:
        """Generate a single response for *prompt*."""
        input_ids: torch.Tensor = cast(
            torch.Tensor,
            self._tokenizer.encode(prompt, return_tensors="pt"),  # type: ignore[no-untyped-call]
        ).to(self._device)

        attention_mask: torch.Tensor = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids: torch.Tensor = cast(
                torch.Tensor,
                self._model.generate(  # type: ignore[no-untyped-call]
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature,
                    top_k=self._config.top_k,
                    top_p=self._config.top_p,
                    repetition_penalty=self._config.repetition_penalty,
                    do_sample=self._config.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,  # type: ignore[arg-type]
                ),
            )

        # Strip the echoed prompt tokens from the output
        generated_ids: torch.Tensor = output_ids[0, input_ids.shape[-1]:]
        response: str = cast(
            str,
            self._tokenizer.decode(  # type: ignore[no-untyped-call]
                generated_ids,
                skip_special_tokens=True,
            ),
        )
        return response.strip()

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                str(self._model_dir),
                use_fast=True,
            ),
        )
        if tokenizer.pad_token is None:  # type: ignore[has-type]
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore[has-type]
        return tokenizer

    def _load_model(self) -> GPT2LMHeadModel:
        model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(  # type: ignore[no-untyped-call]
            str(self._model_dir),
        )
        _ = model.to(self._device)  # type: ignore[no-untyped-call]
        model.eval()
        return model
