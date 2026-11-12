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

#ifndef SCANN_HASHES_CODER_FIXED_POINT_CODER_H_
#define SCANN_HASHES_CODER_FIXED_POINT_CODER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/hashes/coder/coder.h"
#include "scann/proto/coding.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/fixed_point_offline_preprocessing.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/scale_encoding_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

class FixedPointCoder : public Coder {
 public:
  static StatusOr<unique_ptr<FixedPointCoder>> Create(
      const FixedPointCodec& codec) {
    SCANN_RET_CHECK(codec.per_dimension_bits() == 4 ||
                    codec.per_dimension_bits() == 8)
        << codec.per_dimension_bits();
    const ResolvedScaleEncoding resolved_scale_encoding = ResolveScaleEncoding(
        codec.per_dimension_bits(), codec.scale_encoding(), codec.dimension());
    if (resolved_scale_encoding ==
        ResolvedScaleEncoding::kFloat32ScaleBottomBits) {
      SCANN_RET_CHECK_GE(codec.dimension(), kMinDimensionsForBottomBits);
    }

    const ScaleEncoding scale_encoding = codec.scale_encoding();
    FixedPointIndexer indexer;
    switch (codec.fixed_point_method_case()) {
      case FixedPointCodec::kMultiplier:
        indexer = FixedPointIndexer(scale_encoding, codec.per_dimension_bits(),
                                    codec.multiplier(), codec.dimension());
        break;
      case FixedPointCodec::kTrainingResults:
        SCANN_RET_CHECK_EQ(
            codec.dimension(),
            codec.training_results().max_abs_value_by_dim_size());
        indexer = FixedPointIndexer(scale_encoding, codec.per_dimension_bits(),
                                    codec.training_results());
        break;
      default:
        SCANN_RET_CHECK(resolved_scale_encoding !=
                        ResolvedScaleEncoding::kNone);
        indexer = FixedPointIndexer(scale_encoding, codec.per_dimension_bits(),
                                    1.0f, codec.dimension());
    }

    return absl::WrapUnique(new FixedPointCoder(std::move(indexer)));
  }

  absl::Status EncodeDatapoint(absl::Span<const float> original,
                               absl::Span<uint8_t> encoded) const final {
    return indexer_.Hash(original, encoded);
  }

  absl::Status EncodeDatapoint(const DatapointPtr<float>& original_dp,
                               Datapoint<uint8_t>* encoded_dp) const final {
    return indexer_.Hash(original_dp, encoded_dp);
  }

  absl::Status EncodeDatapointWithNoiseShaping(
      absl::Span<const float> original, absl::Span<uint8_t> encoded,
      const double noise_shaping_threshold) const final {
    return indexer_.HashWithNoiseShaping(original, encoded,
                                         noise_shaping_threshold);
  }

  ABSL_DEPRECATED(
      "Set FixedPointCodec.scale_encoding before calling "
      "FixedPointCoder::Create, and use EncodeDatapointWithNoiseShaping().")
  absl::StatusOr<std::string> EncodeScaledDatapointWithNoiseShaping(
      int bits, absl::Span<const float> original, ScaleEncoding scale_encoding,
      const double noise_shaping_threshold) const {
    return indexer_.HashScaledWithNoiseShaping(bits, original, scale_encoding,
                                               noise_shaping_threshold);
  }

  absl::Status DecodeDatapoint(const DatapointPtr<uint8_t>& encoded_dp,
                               Datapoint<float>* decoded_dp) const final {
    return indexer_.Reconstruct(encoded_dp, decoded_dp);
  }

  absl::Status DecodeDatapoint(absl::Span<const uint8_t> encoded,
                               absl::Span<float> decoded) const final {
    return indexer_.Reconstruct(encoded, decoded);
  }

  absl::Status EncodeDatapoint(const DatapointPtr<float>& original_dp,
                               std::string* encoded_str) const final {
    return indexer_.Hash(original_dp, encoded_str);
  }

  absl::Status DecodeDatapoint(absl::string_view encoded_str,
                               Datapoint<float>* decoded_dp) const final {
    return indexer_.Reconstruct(encoded_str, decoded_dp);
  }

  absl::StatusOr<float> DistanceBetweenOriginalAndHashed(
      absl::Span<const float> original, absl::Span<const uint8_t> hashed,
      std::shared_ptr<const DistanceMeasure> distance_override) const final {
    if (!distance_override) {
      distance_override = std::make_shared<DotProductDistance>();
    }
    return indexer_.DistanceBetweenOriginalAndHashed(original, hashed,
                                                     distance_override);
  }

  absl::Status SetSymmetricDistanceMeasure(
      std::shared_ptr<DistanceMeasure> distance) final {
    symmetric_distance_ = std::move(distance);
    return OkStatus();
  }

  absl::StatusOr<float> DistanceBetweenHashed(
      absl::Span<const uint8_t> hashed_1,
      absl::Span<const uint8_t> hashed_2) const final;

  absl::StatusOr<NNResultsVector> AsymmetricScore(
      absl::Span<const float> query,
      std::shared_ptr<DenseDatasetView<uint8_t>> dataset,
      const Coder::ScoringOption& option) const final;

  DatapointIndex original_space_dimension() const final {
    return indexer_.original_space_dimension();
  }

  DatapointIndex hashed_space_bytes() const final {
    return indexer_.hashed_space_bytes();
  }

 private:
  explicit FixedPointCoder(FixedPointIndexer indexer)
      : indexer_(std::move(indexer)) {}

  template <bool kIsCosine>
  absl::StatusOr<NNResultsVector> FixedPointAsymmetricScoreDotOrCosine(
      absl::Span<const float> query,
      const DenseDatasetView<int8_t>* __restrict__ dataset_ptr,
      Coder::ScoringOption option) const;

  FixedPointIndexer indexer_;
  std::shared_ptr<DistanceMeasure> symmetric_distance_;
};

}  // namespace research_scann

#endif
