// Copyright 2026 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_HASHES_CODER_CODER_H_
#define SCANN_HASHES_CODER_CODER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/coding.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/fixed_point_offline_preprocessing.h"
#include "scann/utils/types.h"

namespace research_scann {

class Coder : public VirtualDestructor {
 public:
  virtual absl::Status EncodeDatapoint(absl::Span<const float> original,
                                       absl::Span<uint8_t> encoded) const = 0;

  virtual absl::Status EncodeDatapoint(
      const DatapointPtr<float>& original_dp,
      Datapoint<uint8_t>* encoded_dp) const = 0;

  virtual absl::Status EncodeDatapoint(const DatapointPtr<float>& original_dp,
                                       std::string* encoded_str) const = 0;

  virtual absl::Status EncodeDatapointWithNoiseShaping(
      absl::Span<const float> original, absl::Span<uint8_t> encoded,
      const double noise_shaping_threshold) const = 0;

  virtual absl::Status DecodeDatapoint(absl::Span<const uint8_t> encoded,
                                       absl::Span<float> decoded) const = 0;

  virtual absl::Status DecodeDatapoint(const DatapointPtr<uint8_t>& encoded_dp,
                                       Datapoint<float>* decoded_dp) const = 0;

  virtual absl::Status DecodeDatapoint(absl::string_view encoded_str,
                                       Datapoint<float>* decoded_dp) const = 0;

  virtual absl::StatusOr<float> DistanceBetweenOriginalAndHashed(
      absl::Span<const float> original, absl::Span<const uint8_t> hashed,
      std::shared_ptr<const DistanceMeasure> distance_override) const = 0;

  struct ScoringOption {
    int32_t num_neighbors = 1;

    float epsilon = numeric_limits<float>::infinity();

    bool sort_results = true;

    std::shared_ptr<DistanceMeasure> scoring_distance = nullptr;
  };

  virtual absl::StatusOr<NNResultsVector> AsymmetricScore(
      absl::Span<const float> query,
      std::shared_ptr<DenseDatasetView<uint8_t>> dataset,
      const ScoringOption& option) const = 0;

  virtual absl::Status SetSymmetricDistanceMeasure(
      std::shared_ptr<DistanceMeasure> distance) = 0;

  virtual absl::StatusOr<float> DistanceBetweenHashed(
      absl::Span<const uint8_t> hashed_1,
      absl::Span<const uint8_t> hashed_2) const = 0;

  virtual absl::StatusOr<NNResultsVector> SymmetricScore(
      absl::Span<const uint8_t> query, const DenseDatasetView<uint8_t>& dataset,
      ScoringOption option) const;

  virtual DatapointIndex original_space_dimension() const = 0;

  virtual DatapointIndex hashed_space_bytes() const = 0;
  ABSL_DEPRECATED("Use hashed_space_bytes() instead.")
  DatapointIndex hashed_space_dimension() const { return hashed_space_bytes(); }

  absl::StatusOr<GenericFeatureVector> EncodeGFV(
      const GenericFeatureVector& original_gfv) const;

  absl::StatusOr<GenericFeatureVector> DecodeGFV(
      const GenericFeatureVector& encoded_gfv) const;
};

}  // namespace research_scann

#endif
