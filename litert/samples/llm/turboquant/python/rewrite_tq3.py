#!/usr/bin/env python
"""Phase 3: rewrite model_quantized.tflite -> model_tq3.tflite.

Replaces every externalized-KV attention block with ONE custom op

    voxsum.tq3_attention(q, kv_slice_k, kv_slice_v, mask, packed_k, packed_v)
        -> attn_context

and swaps the full-length fp32 kv_cache_{k,v}_i signature inputs for packed
uint8 TQ3 side-cache inputs packed_{k,v}_i of shape (16384, block_bytes)
(block_bytes = 4-byte fp32 norm + ceil(3*d/8) code bytes: d=256 -> 100,
d=512 -> 196).

Graph facts this tool relies on (audited 2026-07-29):
- prefill_128: 14 attention blocks (layer 14's attention and all KV-shared
  layers are dead code there — no logits output — so kv_cache_{k,v}_14 are
  unused inputs); no transposes; per-block ops:
  BMM(q,Kcache) + BMM(q,Kslice) -> CONCAT -> ADD(mask bcast CONCAT x8)
  -> SOFTMAX -> SLICE x2 -> BMM(P,Vcache) + BMM(P,Vslice) -> ADD.
- decode: 35 blocks. Layers 0-12 read their cache directly; layer 13's cache
  is read through a TRANSPOSE shared by 17 blocks (layer 13 + 16 KV-shared
  sliding layers); layer 14's through a TRANSPOSE shared by 5 blocks (layer 14
  + 4 KV-shared global layers). Same 10-op block shape either way.
- The file uses the extended flatbuffer format (weight buffers after the
  flatbuffer, absolute u64 offset/size). The data section is carried over
  verbatim; offsets are fixed up with a two-pass pack.
"""
import sys
import flatbuffers
from litert_converter import schema_py_generated as s

CACHE_LEN = 16384
N_LAYERS = 15
GLOBAL_EVERY = 5
GLOBAL_DIM = 512
KV_BITS = 3
WINDOW = 0        # >0: sliding layers get WINDOW rows instead of CACHE_LEN

def block_bytes(d):
    if KV_BITS == 16:          # exact fp16: raw storage, no norm, no codes
        return 2 * d
    return 4 + (KV_BITS * d + 7) // 8


def layer_rows(layer):
    """Sliding layers only need their attention window; global layers need the
    full cache. Windowing a global layer would be lossy."""
    is_global = (layer + 1) % GLOBAL_EVERY == 0
    if WINDOW and not is_global and WINDOW < CACHE_LEN:
        return WINDOW
    return CACHE_LEN

def main(inp, outp, cache_len=None, n_layers=None, global_every=None,
         global_dim=None, kv_bits=None, window=None):
    global CACHE_LEN, N_LAYERS, GLOBAL_EVERY, GLOBAL_DIM, KV_BITS, WINDOW
    if kv_bits:
        KV_BITS = int(kv_bits)
    if window:
        WINDOW = int(window)
    if cache_len:
        CACHE_LEN = int(cache_len)
    if n_layers:
        N_LAYERS = int(n_layers)
    if global_every:
        GLOBAL_EVERY = int(global_every)
    if global_dim:
        GLOBAL_DIM = int(global_dim)
    raw = open(inp, "rb").read()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(bytearray(raw), 0))

    offs = [(b.offset, b.size) for b in model.buffers if b.offset and b.offset > 1]
    data_start = min(o for o, _ in offs)
    assert max(o + z for o, z in offs) <= len(raw)
    data = raw[data_start:]

    # Two custom codes because per-instance options cannot reach Run() through
    # the LiteRT custom-op dispatcher (user_data is shared per custom_code):
    #  - voxsum.tq3_attention   : k_new (1,1,T,d), v_new (1,1,d,T)  (direct blocks)
    #  - voxsum.tq3_attention_t : k_new (1,1,d,T), v_new (1,1,T,d)  (KV-shared blocks)
    custom_idx = {}
    for suffix in ("", "_t"):
        oc = s.OperatorCodeT()
        oc.builtinCode = s.BuiltinOperator.CUSTOM
        oc.deprecatedBuiltinCode = s.BuiltinOperator.CUSTOM
        oc.customCode = "voxsum.tq3_attention" + suffix
        oc.version = 1
        custom_idx[suffix] = len(model.operatorCodes)
        model.operatorCodes.append(oc)

    def bcode(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode or 0, c.deprecatedBuiltinCode or 0)

    B = s.BuiltinOperator
    for g in model.subgraphs:
        name = g.name.decode() if isinstance(g.name, bytes) else g.name
        if name not in ("prefill_128", "decode"):
            continue
        tname = {}
        for i, t in enumerate(g.tensors):
            n = t.name.decode() if isinstance(t.name, bytes) else t.name
            tname[i] = n
        byname = {v: k for k, v in tname.items()}
        cons = {}
        prod = {}
        for oi, op in enumerate(g.operators):
            for ti in op.inputs:
                if ti >= 0:
                    cons.setdefault(ti, []).append(oi)
            for ti in op.outputs:
                prod[ti] = oi

        kill = set()
        maybe_dead = set()      # transposes / mask-bcast concats, killed in post-pass
        new_ops = {}            # anchor op index -> [OperatorT, ...]
        removed_inputs = []
        added_inputs = []
        sigmap = {}
        n_blocks = 0

        prefix = name + "_"
        for layer in range(N_LAYERS):
            d = GLOBAL_DIM if (layer + 1) % GLOBAL_EVERY == 0 else 256
            kck = byname[f"{prefix}kv_cache_k_{layer}"]
            kcv = byname[f"{prefix}kv_cache_v_{layer}"]

            packed = []
            for role in ("k", "v"):
                t = s.TensorT()
                t.shape = [layer_rows(layer), block_bytes(d)]
                t.type = s.TensorType.UINT8
                t.buffer = 0
                t.name = f"{prefix}packed_{role}_{layer}"
                g.tensors.append(t)
                packed.append(len(g.tensors) - 1)
            removed_inputs += [kck, kcv]
            added_inputs += packed
            sigmap[f"kv_cache_k_{layer}"] = (f"packed_k_{layer}", packed[0])
            sigmap[f"kv_cache_v_{layer}"] = (f"packed_v_{layer}", packed[1])

            if kck not in cons:
                # prefill: layer 14 attention (and all KV-shared layers) are
                # pruned upstream — only the signature inputs change.
                assert kcv not in cons
                continue

            # resolve score-BMM anchors (direct or via a shared TRANSPOSE)
            (k0,) = cons[kck]
            if bcode(g.operators[k0]) == B.TRANSPOSE:
                maybe_dead.add(k0)
                k_tensor = g.operators[k0].outputs[0]
                anchors = list(cons[k_tensor])
            else:
                k_tensor = kck
                anchors = [k0]
            (v0,) = cons[kcv]
            v_tensors = {kcv}
            if bcode(g.operators[v0]) == B.TRANSPOSE:
                maybe_dead.add(v0)
                v_tensors.add(g.operators[v0].outputs[0])

            for opA in anchors:
                a = g.operators[opA]
                assert bcode(a) == B.BATCH_MATMUL
                q = a.inputs[0] if a.inputs[1] == k_tensor else a.inputs[1]
                (opC,) = cons[a.outputs[0]]
                c = g.operators[opC]
                assert bcode(c) == B.CONCATENATION
                s_new = [t for t in c.inputs if t != a.outputs[0]][0]
                b = g.operators[prod[s_new]]
                assert bcode(b) == B.BATCH_MATMUL
                # the "new tokens" K: kv_slice_k output or its internal alias.
                # Direct blocks carry it as (1,1,T,d); KV-shared blocks (which
                # read the cache through a TRANSPOSE) as (1,1,d,T).
                slice_k = b.inputs[0] if b.inputs[0] != q else b.inputs[1]
                ks = list(g.tensors[slice_k].shape)
                transposed = ks[-1] != d          # T (1 or 128) never equals d
                assert ks[-2 if transposed else -1] == d, (tname[slice_k], ks)
                (opD,) = cons[c.outputs[0]]
                dd = g.operators[opD]
                assert bcode(dd) == B.ADD
                mask_b = [t for t in dd.inputs if t != c.outputs[0]][0]
                opM = prod[mask_b]
                mm = g.operators[opM]
                assert bcode(mm) == B.CONCATENATION
                mask_in = mm.inputs[0]
                assert all(t == mask_in for t in mm.inputs)
                assert "mask" in tname[mask_in], tname[mask_in]
                maybe_dead.add(opM)
                (opE,) = cons[dd.outputs[0]]
                e = g.operators[opE]
                assert bcode(e) == B.SOFTMAX
                opF = opG = None
                for oi in cons[e.outputs[0]]:
                    o = g.operators[oi]
                    assert bcode(o) == B.SLICE
                    if g.tensors[o.outputs[0]].shape[-1] == CACHE_LEN:
                        opF = oi
                    else:
                        opG = oi
                f, g2 = g.operators[opF], g.operators[opG]
                (opFv,) = cons[f.outputs[0]]
                fv = g.operators[opFv]
                assert bcode(fv) == B.BATCH_MATMUL
                assert any(t in v_tensors for t in fv.inputs), \
                    [tname[t] for t in fv.inputs]
                (opGv,) = cons[g2.outputs[0]]
                gv = g.operators[opGv]
                assert bcode(gv) == B.BATCH_MATMUL
                slice_v = [t for t in gv.inputs if t != g2.outputs[0]][0]
                vs = list(g.tensors[slice_v].shape)
                # direct: v_new (d,T); transposed variant: v_new (T,d)
                assert vs[-1 if transposed else -2] == d, (tname[slice_v], vs)
                (opH,) = cons[fv.outputs[0]]
                h = g.operators[opH]
                assert bcode(h) == B.ADD and gv.outputs[0] in h.inputs
                ctx_out = h.outputs[0]

                kill |= {opA, prod[s_new], opC, opD, opE, opF, opG,
                         opFv, opGv, opH}
                op = s.OperatorT()
                op.opcodeIndex = custom_idx["_t" if transposed else ""]
                op.inputs = [q, slice_k, slice_v, mask_in, packed[0], packed[1]]
                op.outputs = [ctx_out]
                op.customOptions = []
                op.customOptionsFormat = 0
                new_ops.setdefault(opA, []).append(op)
                n_blocks += 1

        # post-pass: transposes / mask-broadcast concats whose consumers all died
        for oi in maybe_dead:
            outs = g.operators[oi].outputs
            if all(all(c in kill for c in cons.get(t, [])) for t in outs):
                kill.add(oi)

        ops2 = []
        for oi, op in enumerate(g.operators):
            ops2.extend(new_ops.get(oi, []))
            if oi not in kill:
                ops2.append(op)
        g.operators = ops2
        rm = set(removed_inputs)
        g.inputs = [t for t in g.inputs if t not in rm] + added_inputs

        for sd in model.signatureDefs:
            key = sd.signatureKey.decode() if isinstance(sd.signatureKey, bytes) else sd.signatureKey
            if key != name:
                continue
            ins2 = [tm for tm in sd.inputs
                    if (tm.name.decode() if isinstance(tm.name, bytes) else tm.name)
                    not in sigmap]
            for nm, (pn, ti) in sorted(sigmap.items()):
                tm = s.TensorMapT()
                tm.name = pn
                tm.tensorIndex = ti
                ins2.append(tm)
            sd.inputs = ins2
        print(f"{name}: fused {n_blocks} attention blocks, removed {len(kill)} "
              f"ops, {len(added_inputs)} packed inputs; ops now {len(g.operators)}")

    def pack():
        b = flatbuffers.Builder(64 * 1024 * 1024)
        b.Finish(model.Pack(b), file_identifier=b"TFL3")
        return b.Output()

    fb1 = bytes(pack())
    ALIGN = 64
    new_start = (len(fb1) + ALIGN - 1) // ALIGN * ALIGN
    delta = new_start - data_start
    for buf in model.buffers:
        if buf.offset and buf.offset > 1:
            buf.offset += delta
    fb2 = bytes(pack())
    assert len(fb2) == len(fb1), (len(fb1), len(fb2))
    with open(outp, "wb") as fo:
        fo.write(fb2)
        fo.write(b"\0" * (new_start - len(fb2)))
        fo.write(data)
    print(f"wrote {outp}: flatbuffer {len(fb2)} B + data {len(data)} B "
          f"(delta {delta:+d})")

if __name__ == "__main__":
    main(*sys.argv[1:])
