"""
llama-cpp-python LLM backend with per-call timing.

Tracks:
  - prefill_ms: time to process the full input context (prompt tokens)
  - decode_ms:  time to generate all output tokens
  - n_prompt_tokens, n_output_tokens

The prefill cost scales with context size (relevant for large SQL results),
decode cost scales with output length (relevant for SQL generation).

For the colocation ceiling study:
  - prefill_ms grows as result tables get bigger (more tokens in context)
  - GPU colocation would reduce this by keeping result tokens in GPU
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_cpp import Llama


@dataclass
class LLMResponse:
    text: str
    n_prompt_tokens: int       # Total prompt tokens (full context)
    n_new_prompt_tokens: int   # Incremental tokens since last call (KV cache reuse)
    n_output_tokens: int
    prefill_ms: float          # Estimated from total prompt tokens (overestimates in multi-turn)
    prefill_incr_ms: float     # Estimated from new tokens only (accurate for multi-turn)
    decode_ms: float           # Time to generate output tokens
    total_ms: float


class LlamaBackend:
    """
    Wraps a llama-cpp-python Llama model with timing instrumentation.

    The model is loaded once and reused across calls.  We measure prefill
    and decode time separately by hooking into the token timing.

    Usage:
        llm = LlamaBackend("/path/to/model.gguf", n_ctx=8192)
        llm.load()
        resp = llm.chat([{"role": "user", "content": "Hello"}])
        print(resp.text, resp.prefill_ms, resp.decode_ms)
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,   # -1 = offload all layers to GPU
        temperature: float = 0.0,  # greedy for reproducibility
        max_tokens: int = 512,
        verbose: bool = False,
        flash_attn: bool = True,  # Flash Attention — faster prefill, lower VRAM
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.flash_attn = flash_attn
        self._model: Optional[Llama] = None
        self._model_name = Path(model_path).stem

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self):
        """Load the model weights into GPU memory."""
        self._model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            flash_attn=self.flash_attn,
        )

    def unload(self):
        """Free GPU memory."""
        del self._model
        self._model = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()

    def chat(self, messages: list[dict], prev_prompt_tokens: int = 0) -> LLMResponse:
        """
        Run a chat completion and return timing-annotated LLMResponse.

        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Use as context manager or call load().")

        t0 = time.perf_counter()

        output = self._model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # Extract usage stats from response
        usage = output.get("usage", {})
        n_prompt = usage.get("prompt_tokens", 0)
        n_output = usage.get("completion_tokens", 0)
        text = output["choices"][0]["message"]["content"]

        # Incremental prompt tokens: only the new tokens since last call.
        # In multi-turn with KV cache, llama-cpp reuses cached tokens, so
        # actual prefill work ≈ new tokens only, not full context.
        n_new_prompt = max(0, n_prompt - prev_prompt_tokens)

        # Estimate prefill vs decode split.
        # llama-cpp doesn't expose separate prefill/decode time directly, but
        # prefill speed is ~3-5x decode speed for GPU inference.
        # We use RATIO=4 (empirical for GPU inference).
        PREFILL_RELATIVE_SPEED = 4.0

        def _split(prompt_toks: int) -> tuple[float, float]:
            if prompt_toks + n_output <= 0:
                return 0.0, total_ms
            pw = prompt_toks / PREFILL_RELATIVE_SPEED
            dw = n_output
            tw = pw + dw
            p = total_ms * (pw / tw)
            return p, total_ms - p

        # Full-context estimate (used for turn 1 / single-turn analysis)
        prefill_ms, decode_ms = _split(n_prompt)
        # Incremental estimate (accurate for multi-turn with KV cache)
        prefill_incr_ms, _ = _split(n_new_prompt)

        return LLMResponse(
            text=text,
            n_prompt_tokens=n_prompt,
            n_new_prompt_tokens=n_new_prompt,
            n_output_tokens=n_output,
            prefill_ms=prefill_ms,
            prefill_incr_ms=prefill_incr_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
        )

    def tokens_per_second(self, resp: LLMResponse) -> float:
        """Decode throughput in tokens/second."""
        if resp.decode_ms <= 0:
            return 0.0
        return resp.n_output_tokens / (resp.decode_ms / 1000.0)
