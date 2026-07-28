"""Assemble the VibeVoice-ASR audio front end in PyTorch, from the published checkpoint.

Produces exactly what VibeASR.cpp's C API produces, so the two are directly comparable
and one can replace the other:

    audio @ 24 kHz, [1, n_samples]  ->  [1, n_frames, 1536]

matching `vae_encode_acoustic` + `vae_encode_semantic` followed by prompt_builder's
element-wise sum (utils/prompt_builder.h:215, "acoustic + semantic element-wise sum",
which is the same thing transformers' VibeVoiceAsrMultiModalProjector does).

The checkpoint uses the original VibeVoice tensor names, not the transformers ones —
see remap.py.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from remap import build_encoder_state_dict

from transformers.models.vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerEncoderConfig as EncCfg,
    VibeVoiceAcousticTokenizerEncoderModel as Enc,
)


class Connector(nn.Module):
    """fc1 -> RMSNorm -> fc2, the checkpoint's `*_connector`. Same shape as
    transformers' projector half (linear_1 / norm / linear_2)."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.RMSNorm(out_dim, eps=eps)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class VibeAudioFrontEnd(nn.Module):
    """Both tokenizer encoders and both connectors, summed."""

    def __init__(self, acoustic: Enc, semantic: Enc, a_conn: Connector, s_conn: Connector):
        super().__init__()
        self.acoustic, self.semantic = acoustic, semantic
        self.a_conn, self.s_conn = a_conn, s_conn

    def forward(self, audio):                      # audio [B, n_samples] @ 24 kHz
        x = audio.unsqueeze(1)                     # -> [B, 1, n_samples]
        # Encoders return `latents` already frame-major, [B, n_frames, vae_dim].
        a = self.acoustic(x).latents
        s = self.semantic(x).latents
        return self.a_conn(a) + self.s_conn(s)     # [B, n_frames, 1536]


def _enc_from_cfg(a: dict) -> Enc:
    return Enc(EncCfg(
        channels=a.get("channels", 1),
        hidden_size=a["vae_dim"],
        num_filters=a["encoder_n_filters"],
        downsampling_ratios=tuple(a["encoder_ratios"])[::-1],
        depths=tuple(int(x) for x in a["encoder_depths"].split("-")),
        layer_scale_init_value=a.get("layer_scale_init_value", 1e-6),
        rms_norm_eps=a.get("layernorm_eps", 1e-5),
    ))


def load(model_dir: str) -> VibeAudioFrontEnd:
    d = Path(model_dir)
    cfg = json.load(open(d / "config.json"))
    index = json.load(open(d / "model.safetensors.index.json"))["weight_map"]

    # Open only the shards that actually hold the front end (2 of 3).
    shards = {}
    def get(name):
        f = index[name]
        if f not in shards:
            shards[f] = safe_open(d / f, framework="pt")
        return shards[f].get_tensor(name)

    text_dim = cfg["decoder_config"]["hidden_size"]
    parts = {}
    for tag in ("acoustic", "semantic"):
        enc = _enc_from_cfg(cfg[f"{tag}_tokenizer_config"])
        mapping = build_encoder_state_dict(index.keys(), f"model.{tag}_tokenizer.encoder.")
        enc.load_state_dict({k: get(v) for k, v in mapping.items()}, strict=True)

        conn = Connector(cfg[f"{tag}_vae_dim"], text_dim)
        conn.load_state_dict({
            "fc1.weight": get(f"model.{tag}_connector.fc1.weight"),
            "fc1.bias":   get(f"model.{tag}_connector.fc1.bias"),
            "norm.weight": get(f"model.{tag}_connector.norm.weight"),
            "fc2.weight": get(f"model.{tag}_connector.fc2.weight"),
            "fc2.bias":   get(f"model.{tag}_connector.fc2.bias"),
        }, strict=True)
        parts[tag] = (enc, conn)

    m = VibeAudioFrontEnd(parts["acoustic"][0], parts["semantic"][0],
                          parts["acoustic"][1], parts["semantic"][1])
    return m.eval().float()
