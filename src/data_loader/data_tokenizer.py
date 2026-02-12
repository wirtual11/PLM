from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import unicodedata
import re

@dataclass
class ChunkingConfig:
    tokenizer_name: str = "bert-base-uncased"
    max_tokens: int = 400
    overlap_tokens: int = 60
    lowercase: bool = False
    ascii_punct: bool = True
    max_blank_lines: int = 2
    boundary_aware: bool = True
    # how far back we search for a cleaner boundary in token-space (chars after decode)
    boundary_search_window: int = 220

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    token_count: int
    start_token: int
    end_token: int
    metadata: Dict[str, str] 

class TokenChunker:
    """
    Token-based text chunker for RAG/fine-tuning pipelines.

    Public methods:
      - normalize_text(text): str
      - chunk_text(text, metadata=None): List[TextChunk]
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self._validate_config()
        self.tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained( # type: ignore
                self.config.tokenizer_name,
                use_fast=True,
            ),
        )

    def normalize_text(self, text: str) -> str:
        """
        Normalize raw text for more stable tokenization/retrieval.
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove control chars except newline/tab
        text = "".join(
            ch for ch in text
            if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
        )

        # Whitespace cleanup
        text = text.replace("\t", " ")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(
            r"\n{" + str(self.config.max_blank_lines + 1) + r",}",
            "\n" * self.config.max_blank_lines,
            text,
        )

        if self.config.ascii_punct:
            replacements = {
                "\u2018": "'",
                "\u2019": "'",
                "\u201C": '"',
                "\u201D": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2026": "...",
                "\u00A0": " ",
            }
            for src, dst in replacements.items():
                text = text.replace(src, dst)

        if self.config.lowercase:
            text = text.lower()

        return text.strip()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        End-to-end chunking:
          1) normalize
          2) tokenize
          3) split into overlapping chunks
          4) decode chunk text
        """
        clean = self.normalize_text(text)
        if not clean:
            return []

        token_ids = self._encode(clean)
        if not token_ids:
            return []

        chunks: List[TextChunk] = []
        meta = metadata.copy() if metadata else {}

        start = 0
        chunk_id = 0
        n = len(token_ids)

        while start < n:
            proposed_end = min(start + self.config.max_tokens, n)

            end = proposed_end
            if self.config.boundary_aware and proposed_end < n:
                end = self._find_clean_token_boundary(token_ids, start, proposed_end)

            chunk_ids = token_ids[start:end]
            chunk_text = self._decode(chunk_ids).strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        token_count=len(chunk_ids),
                        start_token=start,
                        end_token=end,
                        metadata=meta.copy(),
                    )
                )
                chunk_id += 1

            if end >= n:
                break

            start = end - self.config.overlap_tokens

        return chunks

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _validate_config(self) -> None:
        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.config.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.config.overlap_tokens >= self.config.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.config.max_blank_lines < 1:
            raise ValueError("max_blank_lines must be >= 1")
        if self.config.boundary_search_window < 40:
            raise ValueError("boundary_search_window is too small")

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _find_clean_token_boundary(
        self,
        token_ids: List[int],
        start: int,
        proposed_end: int,
    ) -> int:
        """
        Try to move chunk end slightly backward to a cleaner textual boundary
        (paragraph/sentence/space) to improve readability + retrieval coherence.
        """
        # Decode only the local region to keep this cheap
        window_start = max(start, proposed_end - self.config.boundary_search_window)
        local_ids = token_ids[window_start:proposed_end]
        local_text = self._decode(local_ids)

        # Search boundary preference order
        candidates = [
            local_text.rfind("\n\n"),
            local_text.rfind(". "),
            local_text.rfind("? "),
            local_text.rfind("! "),
            local_text.rfind("\n"),
            local_text.rfind(" "),
        ]
        best_char_idx = max(candidates)

        # If no useful boundary found, keep proposed end
        if best_char_idx < 0:
            return proposed_end

        # Convert char boundary to token boundary approximately:
        # Re-encode text up to that boundary and map length to token position.
        prefix_text = local_text[: best_char_idx + 1].strip()
        if not prefix_text:
            return proposed_end

        prefix_ids = self._encode(prefix_text)
        candidate_end = window_start + len(prefix_ids)

        # Guardrails: keep reasonable chunk length (>= 60% of max)
        min_len = int(self.config.max_tokens * 0.6)
        if candidate_end - start < min_len:
            return proposed_end
        if candidate_end <= start:
            return proposed_end
        if candidate_end > proposed_end:
            return proposed_end

        return candidate_end
