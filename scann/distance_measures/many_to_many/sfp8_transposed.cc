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

#include "scann/distance_measures/many_to_many/sfp8_transposed.h"

#include <algorithm>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/hashes/coder/fixed_point_coder.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/scale_encoding_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace {

template <typename T, typename Span>
absl::Span<T> CastCharSpan(Span span) {
  static_assert(sizeof(T) == sizeof(char));
  return MakeMutableSpan(reinterpret_cast<T*>(span.data()), span.size());
}

std::unique_ptr<Int8TileCodec> MakeCodec(size_t dims, Int8TileSide side) {
  SCANN_DISPATCH_INT8_TILE(return simd_namespace::NewInt8TileCodec(dims, side));
}

float SquaredL2Norm(ConstSpan<int8_t> dp, float scale) {
  return scale * scale * absl::c_inner_product(dp, dp, 0);
}

}  // namespace

absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>>
SFP8SimdBlockTransposedDatabase::Build(
    const DefaultDenseDatasetView<float>& float_dataset, Int8TileSide side,
    float noise_shaping_threshold) {
  return Build(static_cast<const DenseDatasetView<float>&>(float_dataset), side,
               noise_shaping_threshold);
}

absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>>
SFP8SimdBlockTransposedDatabase::Build(
    const DenseDatasetView<float>& float_dataset, Int8TileSide side,
    float noise_shaping_threshold) {
  FixedPointCodec codec;
  codec.set_dimension(float_dataset.dimensionality());
  codec.set_per_dimension_bits(8);
  codec.set_scale_encoding(ScaleEncoding::FLOAT32_SCALE_SUFFIX);
  SCANN_ASSIGN_OR_RETURN(auto coder, FixedPointCoder::Create(codec));

  DenseDataset<uint8_t> encoded_dataset;
  encoded_dataset.set_dimensionality(coder->hashed_space_bytes());
  encoded_dataset.Resize(float_dataset.size());

  for (DatapointIndex i : IndicesOf(float_dataset)) {
    SCANN_RETURN_IF_ERROR(coder->EncodeDatapointWithNoiseShaping(
        float_dataset.GetDatapointSpan(i), encoded_dataset.mutable_data(i),
        noise_shaping_threshold));
  }
  return Build(codec, DefaultDenseDatasetView<uint8_t>(encoded_dataset), side);
}

absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>>
SFP8SimdBlockTransposedDatabase::Build(
    const FixedPointCodec& codec,
    const DenseDatasetView<uint8_t>& encoded_dataset, Int8TileSide side) {
  SCANN_RET_CHECK_EQ(codec.fixed_point_method_case(),
                     FixedPointCodec::FIXED_POINT_METHOD_NOT_SET);
  SCANN_RET_CHECK_EQ(codec.per_dimension_bits(), 8);
  SCANN_RET_CHECK_NE(codec.scale_encoding(),
                     ScaleEncoding::UNSPECIFIED_SCALE_ENCODING);
  SCANN_ASSIGN_OR_RETURN(
      size_t encoded_bytes,
      ScaledDatapointEncodedBytes(codec.per_dimension_bits(),
                                  codec.scale_encoding(), codec.dimension()));
  SCANN_RET_CHECK_EQ(encoded_bytes, encoded_dataset.dimensionality());
  return std::make_unique<SFP8SimdBlockTransposedDatabase>(
      codec, encoded_dataset, side);
}

SFP8SimdBlockTransposedDatabase::SFP8SimdBlockTransposedDatabase(
    const FixedPointCodec& codec, const DenseDatasetView<uint8_t>& dataset,
    Int8TileSide side)
    : fixed_point_codec_(codec),
      resolved_scale_encoding_(
          ResolveScaleEncoding(8, codec.scale_encoding(), codec.dimension())),
      tile_codec_(MakeCodec(codec.dimension(), side)),
      size_(dataset.size()),

      padded_size_(NextMultipleOf(size_, 2 * tile_codec_->block_datapoints())),
      payload_bytes_(padded_size_ * tile_codec_->datapoint_bytes()),
      hashed_space_bytes_(dataset.dimensionality()),

      payload_(std::make_unique<int8_t[]>(payload_bytes_ +
                                          tile_codec_->register_bytes())),
      scales_(std::make_unique<float[]>(padded_size_)),
      sums_(std::make_unique<int32_t[]>(padded_size_)),
      squared_l2_norms_(std::make_unique<float[]>(padded_size_)) {
  std::fill(scales_.get() + size_, scales_.get() + padded_size_, 0.0f);
  std::fill(sums_.get() + size_, sums_.get() + padded_size_, 0);
  std::fill(squared_l2_norms_.get() + size_,
            squared_l2_norms_.get() + padded_size_, 0.0f);

  auto payload = MutableSpan<int8_t>(payload_.get(), payload_bytes_);
  absl::c_fill(payload, 0);
  for (DatapointIndex dp_idx : Seq(size_)) {
    const auto encoded = dataset.GetDatapointSpan(dp_idx);
    float scale;
    ConstSpan<uint8_t> uint8_dp;
    CHECK_OK(DecodeScaledDatapoint(fixed_point_codec_.per_dimension_bits(),
                                   resolved_scale_encoding_, encoded, scale,
                                   uint8_dp));
    const auto dp = CastCharSpan<const int8_t>(uint8_dp);
    scales_[dp_idx] = scale;
    sums_[dp_idx] = absl::c_accumulate(dp, 0);
    squared_l2_norms_[dp_idx] = SquaredL2Norm(dp, scale);
    tile_codec_->EncodeDatapoint(dp, dp_idx, payload);
  }
}

absl::Status SFP8SimdBlockTransposedDatabase::ReconstructDatapoint(
    DatapointIndex idx, MutableSpan<uint8_t> encoded) const {
  SCANN_RET_CHECK_LT(idx, size_);
  SCANN_RET_CHECK_EQ(encoded.size(), hashed_space_bytes());
  tile_codec_->ReconstructDatapoint(
      idx, payload(),
      CastCharSpan<int8_t>(encoded).subspan(0, dimensionality()));
  if (resolved_scale_encoding_ == ResolvedScaleEncoding::kFloat32ScaleSuffix) {
    absl::little_endian::Store32(encoded.end() - sizeof(uint32_t),
                                 absl::bit_cast<uint32_t>(scales_[idx]));
  }
  return OkStatus();
}

absl::Status SFP8SimdBlockTransposedDatabase::ReconstructFloatDatapoint(
    DatapointIndex idx, MutableSpan<float> float_dp) const {
  SCANN_RET_CHECK_LT(idx, size_);
  SCANN_RET_CHECK_EQ(float_dp.size(), dimensionality());

  tile_codec_->ReconstructFloatDatapoint(idx, payload(), scales_[idx],
                                         float_dp);
  return OkStatus();
}

}  // namespace research_scann
