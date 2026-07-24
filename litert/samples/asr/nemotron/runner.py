#!/usr/bin/env python3
# Copyright 2026. Apache-2.0.
"""Offline transcription with the q4-mix Nemotron LiteRT graphs (RNN-T greedy).

Fully on-device: encoder(INT4) -> prompt_fuse(fp32) -> decoder/joint(fp16) greedy,
four tflite graphs, no torch math at inference (torch here only drives the HF
mel/tokenizer for the desktop demo; on Android those are native). zh-TW output =
the zh-CN slot + OpenCC s2t (the base model has no working zh-TW slot).

zh-TW: point --models at the -zhtw build and use --lang zh-TW; --s2t for
Traditional output, --itn for inverse text normalization (百分之五十->50%).

Usage:
  python -m nemotron.runner --models models/ --wav clip.wav --lang zh-TW --s2t --itn
"""
from __future__ import annotations
import argparse, math, os
import numpy as np, torch, transformers, soundfile as sf
from ai_edge_litert.interpreter import Interpreter

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
BLANK = 13087


class Graph:
    def __init__(self, path):
        self.it = Interpreter(model_path=path); self.it.allocate_tensors()
        self.ins = self.it.get_input_details(); self.outs = self.it.get_output_details()
    def by_shape(self, shape):
        return next(d for d in self.ins if tuple(d["shape"]) == tuple(shape))
    def run(self, feed):
        for d in self.ins:
            self.it.set_tensor(d["index"], feed[d["index"]].astype(d["dtype"]))
        self.it.invoke()
        return [self.it.get_tensor(o["index"]) for o in self.outs]


def transcribe(models, wav, lang, s2t=False, itn=False, max_sym=10):
    proc = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    import json
    slot = json.load(open(os.path.join(
        transformers.utils.hub.cached_file(MODEL_ID, "processor_config.json"))))["prompt_dictionary"][lang] \
        if False else proc(np.zeros(16000, np.float32), sampling_rate=16000, language=lang,
                           return_tensors="pt")["prompt_ids"].item()
    enc = Graph(os.path.join(models, "nemotron_encoder_q4.tflite"))
    fuse = Graph(os.path.join(models, "nemotron_prompt_fuse_fp32.tflite"))  # fp32, on-device
    dec = Graph(os.path.join(models, "nemotron_decoder_fp16.tflite"))
    jnt = Graph(os.path.join(models, "nemotron_joint_fp16.tflite"))
    T_enc = enc.ins[0]["shape"][1]
    n_feat = enc.by_shape((1, T_enc, 128))["index"]
    fz_h = next(d for d in fuse.ins if len(d["shape"]) == 3)["index"]
    fz_oh = fuse.by_shape((1, 128))["index"]
    d_tok = dec.by_shape((1, 1))["index"]
    d_hc = sorted(d["index"] for d in dec.ins if tuple(d["shape"]) == (2, 1, 640))
    j_enc = jnt.by_shape((1, 1, 1024))["index"]; j_dec = jnt.by_shape((1, 1, 640))["index"]

    def dec_step(tok, h, c):
        outs = dec.run({d_tok: np.array([[tok]], np.int32), d_hc[0]: h, d_hc[1]: c})
        do = next(o for o in outs if o.shape == (1, 1, 640))
        th = [o for o in outs if o.shape == (2, 1, 640)]
        return do, th[0], th[1]

    audio, sr = sf.read(wav)
    if audio.ndim > 1: audio = audio.mean(1)
    if sr != 16000:
        import librosa; audio = librosa.resample(np.asarray(audio, np.float32), orig_sr=sr, target_sr=16000)
    feats = proc(np.asarray(audio, np.float32), sampling_rate=16000, language=lang,
                 return_tensors="pt")["input_features"]
    Tc = feats.shape[1]; valid = math.ceil(Tc / 8)
    f = feats.numpy()
    f = np.concatenate([f, np.zeros((1, T_enc - Tc, 128), np.float32)], 1) if Tc < T_enc else f[:, :T_enc]
    hidden = enc.run({n_feat: f})[0]  # [1,Tp,1024] full
    onehot = np.zeros((1, 128), np.float32); onehot[0, slot] = 1.0
    fused = fuse.run({fz_h: hidden.astype(np.float32), fz_oh: onehot})[0][:, :valid]  # trim to valid

    h = np.zeros((2, 1, 640), np.float32); c = np.zeros((2, 1, 640), np.float32)
    do, h, c = dec_step(BLANK, h, c); ids = []
    for t in range(fused.shape[1]):
        ef = fused[:, t:t + 1, :].astype(np.float32)
        for _ in range(max_sym):
            k = int(np.argmax(jnt.run({j_enc: ef, j_dec: do.astype(np.float32)})[0].ravel()))
            if k == BLANK: break
            ids.append(k); do, h, c = dec_step(k, h, c)
    text = proc.batch_decode([torch.tensor(ids)], skip_special_tokens=True)[0] if ids else ""
    if itn:  # inverse text normalization (spoken->written: 百分之五十->50%). deterministic post-proc.
        try:
            from wetext import Normalizer
            text = Normalizer(lang="zh", operator="itn").normalize(text)
        except Exception:
            pass
    if s2t:
        from opencc import OpenCC; text = OpenCC("s2t").convert(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--wav", required=True)
    ap.add_argument("--lang", default="zh-CN"); ap.add_argument("--s2t", action="store_true", help="OpenCC Simplified->Traditional (zh-TW output)")
    ap.add_argument("--itn", action="store_true", help="inverse text normalization (wetext): 百分之五十->50%")
    a = ap.parse_args()
    print(transcribe(a.models, a.wav, a.lang, a.s2t, a.itn))


if __name__ == "__main__":
    main()
