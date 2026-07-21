// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "support/preprocessor/audio_preprocessor_pffft.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_utils.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/io_types.h"
#include "support/util/status_macros.h"
#include "third_party/pffft/sandbox/sandboxed_pffft.sapi.h"
#include "third_party/sandboxed_api/vars.h"

namespace litert::support {

namespace {

// Use Pffft to compute `spectrograms` from `windowed_signals`.
absl::Status ComputeSpectrogram(
    sandboxed_pffft::LibPffftApi& api,
    const std::vector<std::vector<float>>& windowed_signals, int fft_length,
    int fft_bins, std::vector<float>& spectrograms) {
  if (fft_length / 2 + 1 != fft_bins) {
    return absl::InvalidArgumentError("fft_bins must equal fft_length / 2 + 1");
  }

  LITERT_ASSIGN_OR_RETURN(
      sandboxed_pffft::PFFFT_Setup * setup,
      api.pffft_new_setup(fft_length, sandboxed_pffft::PFFFT_REAL));
  if (!setup) {
    return absl::InternalError("Failed to create PFFFT setup.");
  }
  sapi::v::RemotePtr setup_ptr(setup);
  absl::Cleanup destroy_setup = [&]() {
    api.pffft_destroy_setup(&setup_ptr).IgnoreError();
  };

  LITERT_ASSIGN_OR_RETURN(void* scratch_buf,
                          api.pffft_aligned_malloc(fft_length * sizeof(float)));
  sapi::v::RemotePtr scratch_buf_ptr(scratch_buf);
  absl::Cleanup free_scratch = [&]() {
    api.pffft_aligned_free(&scratch_buf_ptr).IgnoreError();
  };

  std::vector<float> data(fft_length);
  sapi::v::Array<float> out_array(data.data(), fft_length);

  spectrograms.reserve(spectrograms.size() +
                       windowed_signals.size() * fft_bins);
  for (const auto& current_window : windowed_signals) {
    sapi::v::Array<const float> in_array(current_window.data(), fft_length);
    LITERT_RETURN_IF_ERROR(api.pffft_transform_ordered(
        &setup_ptr, in_array.PtrBefore(), out_array.PtrAfter(),
        &scratch_buf_ptr, sandboxed_pffft::PFFFT_FORWARD));

    // The DC and half-frequency bins are stashed together as the real and
    // imaginary values in the first output bin.
    float first_bin = data[0];
    spectrograms.push_back(first_bin * first_bin);
    for (size_t i = 2; i < data.size(); i += 2) {
      float real = data[i];
      float imag = data[i + 1];
      spectrograms.push_back(real * real + imag * imag);
    }
    float last_bin = data[1];
    spectrograms.push_back(last_bin * last_bin);
  }

  return absl::OkStatus();
}

}  // namespace

AudioPreprocessorPffft::AudioPreprocessorPffft(
    const AudioPreprocessorConfig& config,
    std::unique_ptr<MelFilterbank> mel_filterbank,
    std::unique_ptr<sandboxed_pffft::LibPffftSandbox> sandbox,
    std::unique_ptr<sandboxed_pffft::LibPffftApi> api)
    : config_(config),
      mel_filterbank_(std::move(mel_filterbank)),
      input_queue_(std::vector<float>()),
      sandbox_(std::move(sandbox)),
      api_(std::move(api)) {
  if (config.GetSemicausalPadding()) {
    samples_to_next_step_ = config.GetFrameLength() - config.GetHopLength();
    input_queue_.resize(config.GetHopLength(), 0.0f);
  } else {
    samples_to_next_step_ = config.GetFrameLength();
  }
}

AudioPreprocessorPffft::~AudioPreprocessorPffft() = default;

AudioPreprocessorPffft::AudioPreprocessorPffft(
    const AudioPreprocessorPffft& other)
    : config_(other.config_),
      mel_filterbank_(nullptr),
      input_queue_(other.input_queue_),
      samples_to_next_step_(other.samples_to_next_step_),
      sandbox_(nullptr),
      api_(nullptr) {
  mel_filterbank_ = std::make_unique<MelFilterbank>();
  auto status = mel_filterbank_->Initialize(
      other.config_.GetFftBins(), other.config_.GetSampleRateHz(),
      other.config_.GetNumMelBins(), other.config_.GetMelLowHz(),
      other.config_.GetMelHighHz());
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize mel filterbank: " << status;
  }
  sandbox_ = std::make_unique<sandboxed_pffft::LibPffftSandbox>();
  status = sandbox_->Init();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize sandboxed pffft: " << status;
  } else {
    api_ = std::make_unique<sandboxed_pffft::LibPffftApi>(sandbox_.get());
  }
}

AudioPreprocessorPffft& AudioPreprocessorPffft::operator=(
    const AudioPreprocessorPffft& other) {
  if (this == &other) {
    return *this;
  }
  config_ = other.config_;
  mel_filterbank_ = std::make_unique<MelFilterbank>();
  auto status = mel_filterbank_->Initialize(
      other.config_.GetFftBins(), other.config_.GetSampleRateHz(),
      other.config_.GetNumMelBins(), other.config_.GetMelLowHz(),
      other.config_.GetMelHighHz());
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize mel filterbank: " << status;
  }
  input_queue_ = other.input_queue_;
  samples_to_next_step_ = other.samples_to_next_step_;
  sandbox_ = std::make_unique<sandboxed_pffft::LibPffftSandbox>();
  status = sandbox_->Init();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize sandboxed pffft: " << status;
  } else {
    api_ = std::make_unique<sandboxed_pffft::LibPffftApi>(sandbox_.get());
  }
  return *this;
}

void AudioPreprocessorPffft::Reset() {
  input_queue_.clear();
  if (config_.GetSemicausalPadding()) {
    samples_to_next_step_ = config_.GetFrameLength() - config_.GetHopLength();
    input_queue_.resize(config_.GetHopLength(), 0.0f);
  } else {
    samples_to_next_step_ = config_.GetFrameLength();
  }
}

absl::StatusOr<std::unique_ptr<AudioPreprocessorPffft>>
AudioPreprocessorPffft::Create(const AudioPreprocessorConfig& config) {
  if (config.GetFrameLength() <= 0) {
    return absl::InvalidArgumentError("Frame length must be positive.");
  }
  auto mel_filterbank = std::make_unique<MelFilterbank>();
  LITERT_RETURN_IF_ERROR(mel_filterbank->Initialize(
      config.GetFftBins(), config.GetSampleRateHz(), config.GetNumMelBins(),
      config.GetMelLowHz(), config.GetMelHighHz()));
  auto sandbox = std::make_unique<sandboxed_pffft::LibPffftSandbox>();
  LITERT_RETURN_IF_ERROR(sandbox->Init());
  auto api = std::make_unique<sandboxed_pffft::LibPffftApi>(sandbox.get());
  return absl::WrapUnique(new AudioPreprocessorPffft(
      config, std::move(mel_filterbank), std::move(sandbox), std::move(api)));
}

absl::Status AudioPreprocessorPffft::PcmFramesToSpectrogram(
    absl::Span<const float> pcm_frames, std::vector<float>& spectrograms) {
  if (!api_) {
    return absl::InternalError("Sandboxed pffft API is not initialized.");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto windowed_signals,
      GetWindowedSignalsForFft(config_, pcm_frames, input_queue_,
                               samples_to_next_step_));
  return ComputeSpectrogram(*api_, windowed_signals, config_.GetFftLength(),
                            config_.GetFftBins(), spectrograms);
}

absl::StatusOr<InputAudio> AudioPreprocessorPffft::Preprocess(
    const InputAudio& input_audio) {
  if (input_audio.IsTensorBuffer()) {
    LITERT_ASSIGN_OR_RETURN(auto processed_audio_tensor,
                            input_audio.GetPreprocessedAudioTensor());
    LITERT_ASSIGN_OR_RETURN(auto processed_audio_tensor_with_reference,
                            processed_audio_tensor->Duplicate());
    InputAudio processed_audio(
        std::move(processed_audio_tensor_with_reference));
    return processed_audio;
  }
  absl::Span<const float> pcm_frames;
  if (input_audio.IsPcmFrames()) {
    LITERT_ASSIGN_OR_RETURN(pcm_frames, input_audio.GetPcmFrames());
  } else {
    return absl::InvalidArgumentError(
        "AudioPreprocessorPffft does not support decoding raw audio bytes; "
        "input must be PCM frames.");
  }

  if (!config_.SkipMelSpectrogramExtraction()) {
    std::vector<float> spectrograms;
    LITERT_RETURN_IF_ERROR(PcmFramesToSpectrogram(pcm_frames, spectrograms));

    std::vector<float> log_mel_spectrograms;
    LITERT_RETURN_IF_ERROR(ToLogMelSpectrogram(
        config_, *mel_filterbank_, spectrograms, log_mel_spectrograms));

    const int num_frames =
        log_mel_spectrograms.size() / config_.GetNumMelBins();
    RankedTensorType mel_tensor_type(
        GetElementType<float>(),
        Layout(Dimensions({1, num_frames, config_.GetNumMelBins()})));
    LITERT_ASSIGN_OR_RETURN(
        auto mel_spectrograms_tensor,
        TensorBuffer::CreateManagedHostMemory(
            mel_tensor_type, log_mel_spectrograms.size() * sizeof(float)));
    LITERT_RETURN_IF_ERROR(mel_spectrograms_tensor.Write<float>(
        absl::MakeSpan(log_mel_spectrograms)));
    return InputAudio(std::move(mel_spectrograms_tensor));
  } else {
    std::vector<float> pcm_vector(pcm_frames.begin(), pcm_frames.end());
    LITERT_ASSIGN_OR_RETURN(auto windowed_signals,
                            GetFramedSegments(config_, pcm_vector, input_queue_,
                                              samples_to_next_step_));

    const int num_frames = windowed_signals.size();
    if (num_frames == 0) {
      return absl::FailedPreconditionError(
          "Not enough samples to form any frame.");
    }
    RankedTensorType mel_tensor_type(
        GetElementType<float>(),
        Layout(Dimensions({1, num_frames, config_.GetFrameLength()})));
    LITERT_ASSIGN_OR_RETURN(
        auto mel_spectrograms_tensor,
        TensorBuffer::CreateManagedHostMemory(
            mel_tensor_type,
            num_frames * config_.GetFrameLength() * sizeof(float)));

    std::vector<float> flat_frames;
    flat_frames.reserve(num_frames * config_.GetFrameLength());
    for (const auto& frame : windowed_signals) {
      flat_frames.insert(flat_frames.end(), frame.begin(), frame.end());
    }
    LITERT_RETURN_IF_ERROR(
        mel_spectrograms_tensor.Write<float>(absl::MakeSpan(flat_frames)));
    return InputAudio(std::move(mel_spectrograms_tensor));
  }
}

}  // namespace litert::support
