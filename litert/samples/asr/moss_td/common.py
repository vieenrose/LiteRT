# Copyright 2026. Apache-2.0.
"""Shared model-building utilities for the MOSS-Transcribe-Diarize LiteRT port.

MOSS-TD 0.9B = Whisper-medium encoder -> 4x time merge -> VQAdaptor(4096->1024)
-> Qwen3-0.6B decoder (audio embeddings masked_scatter'd into the token
embedding sequence).

The port splits the network into three LiteRT models:
  * encoder.tflite  : mel (1,80,3000) -> audio embeddings (1,375,1024)
  * embedder.tflite : token ids -> embeddings, and hidden -> logits (tied)
  * decoder.tflite  : prefill_*/decode signatures over input embeddings with
                      an externalized KV cache and mask-as-input.

The split mirrors the reference C++ implementation (rapidspeech moss_td):
embed lookup + masked_scatter fusion and lm_head are host-side there as well.
It also keeps every flatbuffer below the 2 GB TFLite limit at f32.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import Optional

try:  # torch is only needed for export/verify; the LiteRT runner is torch-free
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - device runners (e.g. Raspberry Pi)
    class _TorchStub:
        @staticmethod
        def inference_mode():
            return lambda f: f

    class _NnStub:
        Module = object

    torch = _TorchStub()
    nn = _NnStub()

DEFAULT_SNAPSHOT_GLOB = (
    "~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/"
    "snapshots/*/"
)

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

AUDIO_TOKEN_ID = 151671
EOS_TOKEN_ID = 151645  # <|im_end|>
PAD_TOKEN_ID = 151643
AUDIO_TOKENS_PER_SECOND = 12.5
AUDIO_MERGE_SIZE = 4
TIME_MARKER_EVERY_SECONDS = 5  # processor_config.json (NOT 2 — verified)
HIDDEN = 1024
MEL_BINS = 80
MEL_FRAMES = 3000
ENC_TOKENS = 375  # 3000 mel frames -> 1500 enc frames -> /4 merge


def resolve_snapshot(path: Optional[str] = None) -> str:
    if path:
        return path
    hits = glob.glob(os.path.expanduser(DEFAULT_SNAPSHOT_GLOB))
    if not hits:
        raise FileNotFoundError(
            "MOSS-Transcribe-Diarize snapshot not found; pass --checkpoint")
    return sorted(hits)[-1]


def load_state_dict(snapshot: str) -> dict[str, torch.Tensor]:
    """Load the MOSS-TD safetensors checkpoint as float32 tensors."""
    from safetensors.torch import load_file

    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = [os.path.basename(p)
                 for p in glob.glob(os.path.join(snapshot, "*.safetensors"))]
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(snapshot, f)))
    return {k: v.float() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Encoder (Whisper-medium encoder + 4x time merge + VQAdaptor)
# ---------------------------------------------------------------------------


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int = 4096, hidden_size: int = HIDDEN,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class MossAudioEncoder(nn.Module):
    """WhisperEncoder + time_merge + VQAdaptor as a single static-shape module.

    Input : mel features (1, 80, 3000)
    Output: audio token embeddings (1, 375, 1024)

    The 4x time merge is group-local, so running it over the padded full
    30 s window and slicing the first `audio_feature_length` tokens on the
    host is exactly equivalent to the reference concat-then-merge.
    """

    def __init__(self, snapshot: str):
        super().__init__()
        from transformers.models.whisper.configuration_whisper import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        with open(os.path.join(snapshot, "config.json")) as f:
            cfg = json.load(f)["audio_config"]
        wcfg = WhisperConfig(**cfg)
        wcfg._attn_implementation = "eager"
        self.encoder = WhisperEncoder(wcfg)
        self.adaptor = VQAdaptor()

        sd = load_state_dict(snapshot)
        enc_sd = {k[len("model.whisper_encoder."):]: v
                  for k, v in sd.items() if k.startswith("model.whisper_encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        missing = [m for m in missing if "embed_positions" not in m]
        assert not missing and not unexpected, (missing, unexpected)
        # embed_positions may be stored explicitly; if present, load it too.
        if "model.whisper_encoder.embed_positions.weight" in sd:
            self.encoder.embed_positions.weight.data.copy_(
                sd["model.whisper_encoder.embed_positions.weight"])
        ad_sd = {k[len("model.vq_adaptor."):]: v
                 for k, v in sd.items() if k.startswith("model.vq_adaptor.")}
        self.adaptor.load_state_dict(ad_sd)
        self.eval()

    @torch.inference_mode()
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_features, return_dict=True).last_hidden_state
        b, t, d = h.shape
        h = h[:, : (t // AUDIO_MERGE_SIZE) * AUDIO_MERGE_SIZE, :]
        h = h.reshape(b, t // AUDIO_MERGE_SIZE, d * AUDIO_MERGE_SIZE)
        return self.adaptor(h)


# ---------------------------------------------------------------------------
# Embedder (tied token embedding + lm_head as host-callable signatures)
# ---------------------------------------------------------------------------


class MossEmbedder(nn.Module):
    """Tied embedding matrix exposed as `embed` and `logits` signatures."""

    def __init__(self, snapshot: str):
        super().__init__()
        sd = load_state_dict(snapshot)
        w = sd["model.language_model.embed_tokens.weight"]  # (vocab, 1024)
        self.weight = nn.Parameter(w, requires_grad=False)
        self.eval()

    @torch.inference_mode()
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(tokens, self.weight)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.weight.t())


class EmbedWrapper(nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, tokens): return self.e.embed(tokens)


class LogitsWrapper(nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, hidden): return self.e.logits(hidden)


# ---------------------------------------------------------------------------
# Decoder body (Qwen3-0.6B blocks + final_norm, embeddings-in / hidden-out)
# ---------------------------------------------------------------------------


def qwen3_06b_config(kv_cache_max_len: int):
    from litert_torch.generative.examples.qwen import qwen3

    cfg = qwen3.get_0_6b_model_config()
    cfg.max_seq_len = max(cfg.max_seq_len, kv_cache_max_len)
    return cfg


class MossDecoderBody(nn.Module):
    """Qwen3-0.6B transformer stack taking input embeddings, emitting the
    final-norm hidden states. No embedding table, no lm_head (see MossEmbedder).

    forward(input_embeds (1,S,1024), input_pos (S,), mask (1,1,S,KV), kv_cache)
      -> {"hidden": (1,S,1024), "kv_cache": updated}
    """

    def __init__(self, snapshot: str, kv_cache_max_len: int):
        super().__init__()
        from litert_torch.generative.layers import attention

        self.config = qwen3_06b_config(kv_cache_max_len)
        self.transformer_blocks = nn.ModuleList(
            attention.TransformerBlock(self.config.block_config(i), self.config)
            for i in range(self.config.num_layers)
        )
        from litert_torch.generative.layers import builder
        self.final_norm = builder.build_norm(
            self.config.embedding_dim, self.config.final_norm_config)

        self._load_weights(snapshot)
        self.eval()

    def _load_weights(self, snapshot: str):
        sd = load_state_dict(snapshot)
        p = "model.language_model."
        new_sd = {}
        for i in range(self.config.num_layers):
            src = f"{p}layers.{i}."
            dst = f"transformer_blocks.{i}."
            new_sd[dst + "atten_func.qkv_projection.weight"] = torch.cat([
                sd[src + "self_attn.q_proj.weight"],
                sd[src + "self_attn.k_proj.weight"],
                sd[src + "self_attn.v_proj.weight"],
            ], dim=0)
            new_sd[dst + "atten_func.output_projection.weight"] = (
                sd[src + "self_attn.o_proj.weight"])
            new_sd[dst + "atten_func.query_norm.weight"] = (
                sd[src + "self_attn.q_norm.weight"])
            new_sd[dst + "atten_func.key_norm.weight"] = (
                sd[src + "self_attn.k_norm.weight"])
            new_sd[dst + "pre_atten_norm.weight"] = (
                sd[src + "input_layernorm.weight"])
            new_sd[dst + "post_atten_norm.weight"] = (
                sd[src + "post_attention_layernorm.weight"])
            new_sd[dst + "ff.w1.weight"] = sd[src + "mlp.gate_proj.weight"]
            new_sd[dst + "ff.w2.weight"] = sd[src + "mlp.down_proj.weight"]
            new_sd[dst + "ff.w3.weight"] = sd[src + "mlp.up_proj.weight"]
        new_sd["final_norm.weight"] = sd[p + "norm.weight"]
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        assert not unexpected, unexpected
        # ff norms are Identity for qwen3 config; nothing else should miss.
        assert not [m for m in missing], missing

    @torch.inference_mode()
    def forward(self, input_embeds: torch.Tensor, input_pos: torch.Tensor,
                mask: torch.Tensor, kv_cache):
        from litert_torch.generative.layers import kv_cache as kv_utils

        attn_config = self.config.block_config(0).attn_config
        n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
        rope = self.config.build_rope(input_pos, n_elem, attn_config.rotary_base)

        x = input_embeds
        updated = []
        for i, block in enumerate(self.transformer_blocks):
            entry = kv_cache.caches[i]
            x, entry = block(x, rope, mask, input_pos, entry)
            updated.append(entry)
        return {
            "hidden": self.final_norm(x),
            "kv_cache": kv_utils.KVCache(tuple(updated)),
        }


# ---------------------------------------------------------------------------
# Prompt building (exact port of processing._audio_span_ids / chat template)
# ---------------------------------------------------------------------------


def audio_span_ids(num_audio_tokens: int, digit_token_ids: dict[str, int]):
    """Interleave audio placeholders with per-2s time-marker digit tokens."""
    n = int(num_audio_tokens)
    if n <= 0:
        return []
    every = TIME_MARKER_EVERY_SECONDS
    tokens_per_marker = int(AUDIO_TOKENS_PER_SECOND * every)
    duration = n / AUDIO_TOKENS_PER_SECOND
    out, consumed = [], 0
    for sec in range(every, int(duration) + 1, every):
        pos = (sec // every) * tokens_per_marker
        seg = pos - consumed
        if seg > 0:
            out.extend([AUDIO_TOKEN_ID] * seg)
            consumed += seg
        out.extend(digit_token_ids[d] for d in str(sec))
    rem = n - consumed
    if rem > 0:
        out.extend([AUDIO_TOKEN_ID] * rem)
    return out


def build_input_ids(tokenizer, num_audio_tokens: int,
                    user_prompt: str = DEFAULT_PROMPT) -> list[int]:
    digit_ids = {}
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        assert len(ids) == 1
        digit_ids[d] = ids[0]
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n"
        + user_prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
    before, after = prompt.split("<|audio_pad|>")
    ids = tokenizer.encode(before, add_special_tokens=False)
    ids += audio_span_ids(num_audio_tokens, digit_ids)
    ids += tokenizer.encode(after, add_special_tokens=False)
    return ids


def chunk_token_lengths(num_samples: int, hop_length: int = 160) -> list[int]:
    """Per-30s-chunk audio token counts: (n-1)//(hop*2*merge)+1."""
    n_samples_per_chunk = 480000
    stride = hop_length * 2 * AUDIO_MERGE_SIZE  # 1280
    out = []
    for start in range(0, num_samples, n_samples_per_chunk):
        n = min(n_samples_per_chunk, num_samples - start)
        out.append((n - 1) // stride + 1)
    return out
