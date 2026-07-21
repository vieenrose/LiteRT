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

#ifndef THIRD_PARTY_ODML_LITERT_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_
#define THIRD_PARTY_ODML_LITERT_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/io_types.h"

namespace sandboxed_pffft {
class LibPffftSandbox;
class LibPffftApi;
}  // namespace sandboxed_pffft

namespace litert::support {

// Audio preprocessor implementation using sandboxed PFFFT library.
class AudioPreprocessorPffft : public AudioPreprocessor {
 public:
  ~AudioPreprocessorPffft() override;

  // Creates an AudioPreprocessorPffft instance.
  // Args:
  //   - config: The configuration of the audio preprocessor.
  // Returns:
  //   A unique pointer to the AudioPreprocessorPffft instance.
  static absl::StatusOr<std::unique_ptr<AudioPreprocessorPffft>> Create(
      const AudioPreprocessorConfig& config);

  // Preprocesses the audio (PCM frames or pre-processed audio) and returns the
  // preprocessed audio mel spectrograms.
  absl::StatusOr<InputAudio> Preprocess(const InputAudio& input_audio) override;

  // Resets the preprocessor to its initial state.
  void Reset() override;

  // Copy constructor for cloning the audio preprocessor.
  AudioPreprocessorPffft(const AudioPreprocessorPffft& other);

  // Copy assignment operator for cloning the audio preprocessor.
  AudioPreprocessorPffft& operator=(const AudioPreprocessorPffft& other);

 private:
  explicit AudioPreprocessorPffft(
      const AudioPreprocessorConfig& config,
      std::unique_ptr<MelFilterbank> mel_filterbank,
      std::unique_ptr<sandboxed_pffft::LibPffftSandbox> sandbox,
      std::unique_ptr<sandboxed_pffft::LibPffftApi> api);

  absl::Status PcmFramesToSpectrogram(absl::Span<const float> pcm_frames,
                                      std::vector<float>& spectrograms);

  AudioPreprocessorConfig config_;
  std::unique_ptr<MelFilterbank> mel_filterbank_;
  std::vector<float> input_queue_;
  int samples_to_next_step_;
  std::unique_ptr<sandboxed_pffft::LibPffftSandbox> sandbox_;
  std::unique_ptr<sandboxed_pffft::LibPffftApi> api_;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_
