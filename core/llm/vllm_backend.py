"""
vLLM backend for production-grade LLM inference on L40S / A100 / H100.

Key differences from llama_backend.py:
  - Uses HuggingFace model names (not GGUF), supports BF16 / FP8 / AWQ / GPTQ
  - Flash Attention enabled automatically for Ada Lovelace (L40S) and Hopper (H100)
  - Prefix caching: KV pairs for repeated prefixes (schema, system prompt) reused
    across calls — this is what our incr_prefill metric approximates, but here
    it actually happens in hardware
  - Real TTFT timing via RequestOutput.metrics (no PREFILL_RELATIVE_SPEED estimate)
    TTFT = time to first token = true prefill wall-clock time
    decode_ms = total_ms - TTFT = true decode wall-clock time

Supported quantizations (set via dtype / quantization args):
  - "auto"   : BF16 on Ampere+, FP16 on older
  - "fp8"    : L40S / H100 FP8 (fastest, ~2x vs BF16)
  - "awq"    : load AWQ pre-quantized model (e.g. Qwen2.5-32B-Instruct-AWQ)
  - "gptq"   : load GPTQ pre-quantized model

Model VRAM budget on L40S (48GB):
  Qwen2.5-7B-Instruct  (BF16)  ~15GB  — fits easily
  Qwen2.5-14B-Instruct (BF16)  ~29GB  — fits, ~19GB left for KV cache
  Qwen2.5-32B-Instruct (AWQ)   ~20GB  — fits, ~28GB left for KV cache
  Qwen2.5-72B-Instruct         —      — needs 2× L40S
"""

import time
from dataclasses import dataclass
from typing import Optional

from .llama_backend import LLMResponse  # reuse the same response dataclass


class VLLMBackend:
    """
    Production LLM backend wrapping vLLM.

    Matches the LlamaBackend interface so all experiments work unchanged.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.
        e.g. "Qwen/Qwen2.5-7B-Instruct" or "/data/models/qwen2.5-14b"
    dtype : str
        "auto" (BF16 on Ampere+), "float16", "bfloat16", "fp8"
    quantization : str | None
        None, "awq", "gptq", "fp8"
    max_model_len : int
        Max context window. Reduce to leave more VRAM for KV cache.
    gpu_memory_utilization : float
        Fraction of GPU memory vLLM may use (default 0.90).
    enable_prefix_caching : bool
        Cache KV pairs for repeated prefixes across requests.
        Critical for multi-turn agents sharing the same system prompt + schema.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism (1 for single L40S).
    temperature : float
        0.0 = greedy (reproducible).
    max_tokens : int
        Max output tokens per call.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.dtype = dtype
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return self.model.split("/")[-1]

    def load(self):
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel

        kwargs = dict(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization

        self._llm = LLM(**kwargs)
        # Pre-build the sampling params object (reused per call)
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def unload(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
            # Release GPU memory
            import torch
            torch.cuda.empty_cache()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()

    def chat(self, messages: list[dict], prev_prompt_tokens: int = 0) -> LLMResponse:
        """
        Run a chat completion and return timing-annotated LLMResponse.

        Uses vLLM's RequestOutput.metrics for real TTFT (no estimation).
        TTFT = prefill wall-clock time (includes prefix cache lookup).
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Use as context manager or call load().")

        # Apply chat template to convert messages → prompt string
        from vllm import SamplingParams
        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        t_submit = time.perf_counter()
        outputs = self._llm.generate(
            [prompt],
            self._sampling_params,
        )
        t_done = time.perf_counter()

        output = outputs[0]
        text = output.outputs[0].text
        n_prompt = output.metrics.num_cached_tokens + len(output.prompt_token_ids) \
                   if hasattr(output.metrics, 'num_cached_tokens') \
                   else len(output.prompt_token_ids)
        n_prompt = len(output.prompt_token_ids)
        n_output = len(output.outputs[0].token_ids)

        # Real TTFT from vLLM metrics
        metrics = output.metrics
        total_ms = (t_done - t_submit) * 1000.0

        if (hasattr(metrics, 'first_token_time') and
                metrics.first_token_time is not None and
                metrics.arrival_time is not None):
            prefill_ms  = (metrics.first_token_time - metrics.arrival_time) * 1000.0
            decode_ms   = total_ms - prefill_ms
        else:
            # Fallback if metrics not available
            PREFILL_RELATIVE_SPEED = 4.0
            if n_prompt + n_output > 0:
                pw = n_prompt / PREFILL_RELATIVE_SPEED
                prefill_ms = total_ms * (pw / (pw + n_output))
            else:
                prefill_ms = 0.0
            decode_ms = total_ms - prefill_ms

        # Incremental tokens since last call (KV cache reuse)
        n_new_prompt = max(0, n_prompt - prev_prompt_tokens)
        # Scale prefill_ms by new/total token fraction for incr estimate
        prefill_incr_ms = prefill_ms * (n_new_prompt / n_prompt) if n_prompt > 0 else prefill_ms

        # Note: with prefix caching enabled, prefill_ms already reflects cache hits.
        # On a cache hit, vLLM skips attention for cached tokens, so TTFT is much
        # lower than a full prefill — this is the hardware-level version of incr_prefill.

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
