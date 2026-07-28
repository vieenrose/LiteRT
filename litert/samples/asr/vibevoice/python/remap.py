"""Map VibeVoice-ASR-BitNet checkpoint tensor names onto the transformers module names.

The published checkpoint predates the transformers integration, and
`_checkpoint_conversion_mapping` is None on VibeVoiceAsrModel, so nothing does this
for us. The two layouts describe the same graph with different nesting:

  checkpoint                                   transformers
  --------------------------------------------------------------------------
  downsample_layers.0.0.conv.conv.W       ->   stem.conv.conv.W
  downsample_layers.{i}.0.conv.conv.W     ->   conv_layers.{i-1}.conv.conv.W
  stages.0.{j}.<rest>                     ->   stem.stage.{j}.<rest>
  stages.{i}.{j}.<rest>                   ->   conv_layers.{i-1}.stage.{j}.<rest>
  head.conv.conv.W                        ->   head.conv.W
  ...mixer.conv.conv.conv.W               ->   ...mixer.conv.W

i.e. transformers folds stage 0 (the stride-1 full-resolution stage) into a `stem`
and pairs each later downsample with its stage inside one `conv_layers` entry.
"""

import re


def remap_encoder_key(k: str) -> str:
    """Checkpoint-relative encoder key -> transformers encoder state_dict key."""
    # Depthwise mixer carries two redundant .conv levels in the checkpoint.
    k = k.replace("mixer.conv.conv.conv.", "mixer.conv.")
    # The head has one.
    k = k.replace("head.conv.conv.", "head.conv.")

    m = re.match(r"downsample_layers\.(\d+)\.0\.(.*)$", k)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        return f"stem.{rest}" if i == 0 else f"conv_layers.{i - 1}.{rest}"

    m = re.match(r"stages\.(\d+)\.(\d+)\.(.*)$", k)
    if m:
        i, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"stem.stage.{j}.{rest}" if i == 0 else f"conv_layers.{i - 1}.stage.{j}.{rest}"

    return k


def build_encoder_state_dict(ckpt_keys, prefix):
    """{transformers key: checkpoint key} for one encoder under `prefix`."""
    out = {}
    for full in ckpt_keys:
        if not full.startswith(prefix):
            continue
        out[remap_encoder_key(full[len(prefix):])] = full
    return out
