#!/system/bin/sh
# On-device benchmark for the MOSS-TD LiteRT port (Samsung SM-A5360, arm64).
# Usage: sh phone_bench.sh <variant>   (variant: q8 | fp16)
# Expects in /data/local/tmp/mosslite:
#   android_aarch64_benchmark_model, moss_td_{encoder,embedder}_<v>.tflite,
#   moss_td_decoder_<v>_ekv2048.tflite
set -e
cd /data/local/tmp/mosslite
V=$1
BM=./android_aarch64_benchmark_model
COMMON="--num_threads=8 --use_xnnpack=true --report_peak_memory_footprint=true"

echo "=== encoder $V ==="
$BM --graph=moss_td_encoder_$V.tflite $COMMON --num_runs=5 2>&1 | \
  grep -E "Inference timings|memory footprint|INFO: .*avg"

echo "=== embedder $V (logits sig) ==="
$BM --graph=moss_td_embedder_$V.tflite --signature_to_run_for=logits \
  $COMMON --num_runs=20 2>&1 | grep -E "Inference timings|memory footprint"

echo "=== embedder $V (embed_1 sig) ==="
$BM --graph=moss_td_embedder_$V.tflite --signature_to_run_for=embed_1 \
  $COMMON --num_runs=20 2>&1 | grep -E "Inference timings|memory footprint"

echo "=== decoder $V decode sig ==="
$BM --graph=moss_td_decoder_${V}_ekv2048.tflite --signature_to_run_for=decode \
  $COMMON --num_runs=20 2>&1 | grep -E "Inference timings|memory footprint"

echo "=== decoder $V prefill_128 sig ==="
$BM --graph=moss_td_decoder_${V}_ekv2048.tflite \
  --signature_to_run_for=prefill_128 $COMMON --num_runs=3 2>&1 | \
  grep -E "Inference timings|memory footprint"

echo "=== decoder $V prefill_1024 sig ==="
$BM --graph=moss_td_decoder_${V}_ekv2048.tflite \
  --signature_to_run_for=prefill_1024 $COMMON --num_runs=2 2>&1 | \
  grep -E "Inference timings|memory footprint"
