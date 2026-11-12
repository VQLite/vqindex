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



#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_SFP8_TRANSPOSED_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_SFP8_TRANSPOSED_H_

#include <cstdint>
#include <ostream>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/many_to_many/int8_tile.h"
#include "scann/distance_measures/one_to_many/resolved_scale_encoding.h"
#include "scann/proto/coding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

namespace research_scann {

class SFP8SimdBlockTransposedDatabase {
 public:
  static absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>> Build(
      const DefaultDenseDatasetView<float>& float_dataset, Int8TileSide side,
      float noise_shaping_threshold = NAN);
  static absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>> Build(
      const DenseDatasetView<float>& float_dataset, Int8TileSide side,
      float noise_shaping_threshold = NAN);

  static absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>> Build(
      const FixedPointCodec& codec,
      const DenseDatasetView<uint8_t>& encoded_dataset, Int8TileSide side);

  PlatformGeneration platform_generation() const {
    return tile_codec_->platform_generation();
  }

  DimensionIndex dimensionality() const {
    return tile_codec_->dimensionality();
  }

  DimensionIndex hashed_space_bytes() const { return hashed_space_bytes_; }

  absl::Status ReconstructDatapoint(DatapointIndex idx,
                                    MutableSpan<uint8_t> encoded) const;

  absl::Status ReconstructFloatDatapoint(DatapointIndex idx,
                                         MutableSpan<float> float_dp) const;

  Int8TileSide side() const { return tile_codec_->side(); }

  DatapointIndex size() const { return size_; }

  inline bool empty() const { return size_ == 0; }

  size_t datapoint_bytes() const { return tile_codec_->datapoint_bytes(); }

  size_t block_bytes() const { return tile_codec_->block_bytes(); }

  ConstSpan<int8_t> payload() const { return {payload_.get(), payload_bytes_}; }

  ConstSpan<float> scales() const { return {scales_.get(), size_}; }

  ConstSpan<int32_t> sums() const { return {sums_.get(), size_}; }

  ConstSpan<float> squared_l2_norms() const {
    return {squared_l2_norms_.get(), size_};
  }

  SFP8SimdBlockTransposedDatabase(
      const FixedPointCodec& codec,
      const DenseDatasetView<uint8_t>& encoded_dataset, Int8TileSide side);

 private:
  const FixedPointCodec fixed_point_codec_;
  const ResolvedScaleEncoding resolved_scale_encoding_;
  const std::unique_ptr<Int8TileCodec> tile_codec_;

  const DatapointIndex size_;
  const size_t padded_size_;

  const size_t payload_bytes_;

  const DatapointIndex hashed_space_bytes_;

  unique_ptr<int8_t[]> payload_;

  unique_ptr<float[]> scales_;

  unique_ptr<int32_t[]> sums_;

  unique_ptr<float[]> squared_l2_norms_;

  friend std::ostream& operator<<(std::ostream& os,
                                  const SFP8SimdBlockTransposedDatabase& db);
};

inline std::ostream& operator<<(std::ostream& os,
                                const SFP8SimdBlockTransposedDatabase& db) {
  return os << "SFP8SimdBlockTransposedDatabase{tile_codec_=" << *db.tile_codec_
            << ", size_=" << db.size_ << ", padded_size_=" << db.padded_size_
            << ", payload_bytes_=" << db.payload_bytes_
            << ", hashed_space_bytes_=" << db.hashed_space_bytes_
            << ", fixed_point_codec_=" << db.fixed_point_codec_.DebugString()
            << "}";
}

}  // namespace research_scann

#endif
