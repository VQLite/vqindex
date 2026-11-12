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



#include "scann/utils/fixed_point/fixed_point_offline_preprocessing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/fixed_point_training_results.pb.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/scale_encoding_helpers.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {

FixedPointTrainer::~FixedPointTrainer() {}

unique_ptr<FixedPointTrainer> FixedPointTrainer::New(
    float multiplier_quantile, DatapointIndex num_datapoints) {
  CHECK_LE(multiplier_quantile, 1.0) << "Multiplier quantile must be <= 1.0.";
  CHECK_GT(multiplier_quantile, 0.0) << "Multiplier quantile must be > 0.0.";
  if (multiplier_quantile < 1.0) {
    return std::make_unique<FixedPointQuantileTrainer>(multiplier_quantile,
                                                       num_datapoints);
  } else {
    return std::make_unique<FixedPointMaxTrainer>();
  }
}

absl::Status FixedPointMaxTrainer::AddDatapoint(
    const DatapointPtr<float>& dptr) {
  if (!dptr.IsDense()) {
    return InvalidArgumentError(
        "FixedPointTrainer::AddDatapoint may only be called with dense "
        "datapoints.");
  }
  if (results_extracted_) {
    return FailedPreconditionError(
        "Cannot call FixedPointTrainer::AddDatapoint after ExtractResults has "
        "already been called.");
  }
  if (add_datapoint_called_at_least_once_) {
    if (dptr.dimensionality() != results_.max_abs_value_by_dim_size()) {
      return InvalidArgumentError(
          "Mismatch between dptr.dimensionality and dimensionality of "
          "previously added datapoints.  (%d vs. %d).",
          dptr.dimensionality(), results_.max_abs_value_by_dim_size());
    }
  } else {
    add_datapoint_called_at_least_once_ = true;
    results_.mutable_max_abs_value_by_dim()->Resize(dptr.dimensionality(),
                                                    0.0f);
  }
  for (size_t i = 0; i < dptr.dimensionality(); ++i) {
    results_.set_max_abs_value_by_dim(
        i,
        std::max(results_.max_abs_value_by_dim(i), std::abs(dptr.values()[i])));
  }
  return OkStatus();
}

FixedPointTrainingResults FixedPointMaxTrainer::ExtractResults() {
  results_extracted_ = true;
  return results_;
}

FixedPointQuantileTrainer::FixedPointQuantileTrainer(
    float multiplier_quantile, DatapointIndex num_datapoints)
    : num_datapoints_(num_datapoints),
      multiplier_quantile_(multiplier_quantile) {
  CHECK_LE(multiplier_quantile, 1.0) << "Multiplier quantile must be <= 1.0.";
  CHECK_GT(multiplier_quantile, 0.0) << "Multiplier quantile must be > 0.0.";
}

absl::Status FixedPointQuantileTrainer::AddDatapoint(
    const DatapointPtr<float>& dptr) {
  if (!dptr.IsDense()) {
    return InvalidArgumentError(
        "FixedPointTrainer::AddDatapoint may only be called with dense "
        "datapoints.");
  }
  if (num_datapoints_added_ > 0) {
    if (dptr.dimensionality() != top_ns_.size()) {
      return InvalidArgumentError(
          "Mismatch between dptr.dimensionality and dimensionality of "
          "previously added datapoints.  (%d vs. %d).",
          dptr.dimensionality(), top_ns_.size());
    }
  }
  if (num_datapoints_added_ >= num_datapoints_) {
    return FailedPreconditionError(
        "Cannot add a %dth datapoint to a FixedPointQuantileTrainer intended "
        "for %d datapoints.",
        num_datapoints_added_ + 1, num_datapoints_);
  }
  if (num_datapoints_added_ == 0) {
    const DatapointIndex num_datapoints_to_keep =
        num_datapoints_ * (1.0 - multiplier_quantile_) + 1;
    top_ns_ = vector<TopNAmortizedConstant<float>>(dptr.dimensionality());
    for (auto& elem : top_ns_) {
      elem = TopNAmortizedConstant<float>(num_datapoints_to_keep);
    }
  }
  ++num_datapoints_added_;
  for (size_t i = 0; i < dptr.dimensionality(); ++i) {
    top_ns_[i].push(std::abs(dptr.values()[i]));
  }
  return OkStatus();
}

FixedPointTrainingResults FixedPointQuantileTrainer::ExtractResults() {
  FixedPointTrainingResults result;
  result.mutable_max_abs_value_by_dim()->Reserve(top_ns_.size());
  for (auto& top_n : top_ns_) {
    result.add_max_abs_value_by_dim(top_n.exact_bottom());
  }
  return result;
}

FixedPointGfvConverter::FixedPointGfvConverter(
    const FixedPointTrainingResults& training_results, Distance distance,
    double noise_shaping_threshold)
    : distance_(distance), noise_shaping_threshold_(noise_shaping_threshold) {
  if (!std::isnan(noise_shaping_threshold_)) {
    QCHECK_GE(noise_shaping_threshold_, 0.0);
  }
  multipliers_.reserve(training_results.max_abs_value_by_dim_size());
  for (float f : training_results.max_abs_value_by_dim()) {
    multipliers_.push_back(numeric_limits<int8_t>::max() / f);
  }
}

FixedPointGfvConverter::FixedPointGfvConverter(
    vector<float> per_dimension_multipliers, Distance distance)
    : distance_(distance), multipliers_(std::move(per_dimension_multipliers)) {}

FixedPointGfvConverter::FixedPointGfvConverter(float multiplier,
                                               Distance distance,
                                               double noise_shaping_threshold)
    : distance_(distance),
      using_per_dimension_multipliers_(false),
      universal_multiplier_(multiplier),
      noise_shaping_threshold_(noise_shaping_threshold) {}

absl::Status FixedPointGfvConverter::Convert(GenericFeatureVector* gfv) const {
  DCHECK(gfv);
  if (gfv->feature_value_float().empty()) {
    return InvalidArgumentError(
        "FixedPointGfvConverter::Convert only works with FLOAT GFVs.");
  }
  if (using_per_dimension_multipliers_ &&
      gfv->feature_value_float_size() != multipliers_.size()) {
    return InvalidArgumentError(
        StrFormat("Dimensionality mismatch (GFV dimensionality = %d, expected "
                  "dimensionality = %d).",
                  gfv->feature_value_float_size(), multipliers_.size()));
  }
  if (!gfv->feature_index().empty()) {
    return InvalidArgumentError(
        "FixedPointGfvConverter only works for dense GFVs.");
  }
  if (gfv->feature_value_int64_size() > 0) {
    return InvalidArgumentError(
        "FixedPointGfvConverter cannot overwrite pre-existing "
        "feature_value_int64 field.");
  }

  gfv->mutable_feature_value_int64()->Reserve(gfv->feature_value_float_size());
  if (using_per_dimension_multipliers_) {
    if (std::isnan(noise_shaping_threshold_)) {
      for (size_t i = 0; i < gfv->feature_value_float_size(); ++i) {
        gfv->add_feature_value_int64(
            Int8Quantize(gfv->feature_value_float(i) * multipliers_[i]));
      }
    } else {
      std::vector<int8_t> uint8_quantized(gfv->feature_value_float_size());
      ScalarQuantizeFloatDatapointWithNoiseShaping(
          MakeDatapointPtr(gfv->feature_value_float()), multipliers_,
          noise_shaping_threshold_, &uint8_quantized);
      for (const int8_t q : uint8_quantized) {
        gfv->add_feature_value_int64(q);
      }
    }
  } else {
    if (std::isnan(noise_shaping_threshold_)) {
      for (size_t i = 0; i < gfv->feature_value_float_size(); ++i) {
        gfv->add_feature_value_int64(
            Int8Quantize(gfv->feature_value_float(i) * universal_multiplier_));
      }
    } else {
      vector<float> multipliers(gfv->feature_value_float_size(),
                                universal_multiplier_);
      std::vector<int8_t> uint8_quantized(gfv->feature_value_float_size());
      ScalarQuantizeFloatDatapointWithNoiseShaping(
          MakeDatapointPtr(gfv->feature_value_float()), multipliers,
          noise_shaping_threshold_, &uint8_quantized);
      for (const int8_t q : uint8_quantized) {
        gfv->add_feature_value_int64(q);
      }
    }
  }
  if (distance_ == ALL || distance_ == SQUARED_L2) {
    gfv->clear_fixed_point_metadata();
    gfv->mutable_fixed_point_metadata()->set_squared_l2_norm(
        SquaredL2Norm(MakeDatapointPtr(gfv->feature_value_float().data(),
                                       gfv->feature_value_float_size())));
  }
  gfv->clear_feature_value_float();
  gfv->set_feature_type(GenericFeatureVector::INT64);
  return OkStatus();
}

absl::Status FixedPointIndexer::Hash(ConstSpan<float> original,
                                     MutableSpan<uint8_t> encoded) const {
  return HashWithNoiseShaping(original, encoded, NAN);
}

absl::Status FixedPointIndexer::Hash(const DatapointPtr<float>& original_dp,
                                     Datapoint<uint8_t>* encoded_dp) const {
  if (!original_dp.IsDense()) {
    return InvalidArgumentError(
        "FixedPointIndexer::Hash may only be called with dense "
        "datapoints.");
  }
  encoded_dp->clear();
  encoded_dp->mutable_values()->resize(original_dp.dimensionality());
  return Hash(original_dp.values_span(), encoded_dp->mutable_values_span());
}

absl::Status FixedPointIndexer::Hash(const DatapointPtr<float>& original_dp,
                                     std::string* encoded_str) const {
  if (!original_dp.IsDense()) {
    return InvalidArgumentError(
        "FixedPointIndexer::Hash may only be called with dense "
        "datapoints.");
  }
  encoded_str->resize(hashed_space_bytes());
  auto mutable_span = MakeMutableSpan(
      reinterpret_cast<uint8_t*>(const_cast<char*>(encoded_str->data())),
      hashed_space_bytes());
  return Hash(original_dp.values_span(), mutable_span);
}

absl::Status FixedPointIndexer::HashWithNoiseShaping(
    ConstSpan<float> original, MutableSpan<uint8_t> encoded,
    const double noise_shaping_threshold) const {
  SCANN_RET_CHECK_EQ(original.size(), original_space_dimension());
  SCANN_RET_CHECK_EQ(encoded.size(), hashed_space_bytes());
  ConstSpan<float> multipliers =
      using_per_dimension_multipliers_
          ? multipliers_
          : ConstSpan<float>(&universal_multiplier_, 1);
  const DatapointPtr<float> dptr = MakeDatapointPtr(original);
  return QuantizeScaledFloatDatapointWithNoiseShaping(
      per_dimension_bits_, dptr, multipliers, scale_encoding_,
      noise_shaping_threshold, encoded);
}

absl::StatusOr<std::string> FixedPointIndexer::HashScaledWithNoiseShaping(
    int bits, absl::Span<const float> original, ScaleEncoding scale_encoding,
    const double noise_shaping_threshold) const {
  std::string result;
  if (!using_per_dimension_multipliers_) {
    return InternalError(
        "HashScaledWithNoiseShaping can only be used with per dimension "
        "multipliers");
  }
  const DatapointPtr<float> dptr = MakeDatapointPtr(original);
  SCANN_RETURN_IF_ERROR(AppendQuantizeScaledFloatDatapointWithNoiseShaping(
      bits, dptr, multipliers_, scale_encoding, noise_shaping_threshold,
      result));
  return result;
}

absl::Status FixedPointIndexer::Reconstruct(
    ConstSpan<uint8_t> encoded, MutableSpan<float> reconstructed) const {
  SCANN_RET_CHECK_EQ(reconstructed.size(), original_space_dimension());
  SCANN_RET_CHECK_EQ(encoded.size(), hashed_space_bytes());
  ConstSpan<float> inverse_fixed8_multipliers =
      using_per_dimension_multipliers_
          ? inv_multipliers_
          : ConstSpan<float>(&universal_inv_multiplier_, 1);
  return ReconstructScaledDatapoint(per_dimension_bits_,
                                    inverse_fixed8_multipliers, scale_encoding_,
                                    encoded, reconstructed);
}

absl::StatusOr<float> FixedPointIndexer::DistanceBetweenOriginalAndHashed(
    absl::Span<const float> original, absl::Span<const uint8_t> hashed,
    std::shared_ptr<const DistanceMeasure> distance_override) const {
  SCANN_RET_CHECK_EQ(original.size(), original_space_dimension());
  SCANN_RET_CHECK_EQ(hashed.size(), hashed_space_bytes());
  if (using_per_dimension_multipliers_ && per_dimension_bits_ == 8 &&
      scale_encoding_ == UNSPECIFIED_SCALE_ENCODING) {
    if (distance_override->specially_optimized_distance_tag() ==
        DistanceMeasure::DOT_PRODUCT) {
      return -DenseDotProduct(
          MakeDatapointPtr(reinterpret_cast<const int8_t*>(hashed.data()),
                           hashed.size()),
          MakeDatapointPtr(inv_multipliers_), MakeDatapointPtr(original));
    } else if (distance_override->specially_optimized_distance_tag() ==
               DistanceMeasure::COSINE) {
      return 1.0 -
             DenseDotProduct(MakeDatapointPtr(
                                 reinterpret_cast<const int8_t*>(hashed.data()),
                                 hashed.size()),
                             MakeDatapointPtr(inv_multipliers_),
                             MakeDatapointPtr(original));
    }
  }
  std::vector<float> reconstruced(original.size());
  SCANN_RETURN_IF_ERROR(Reconstruct(hashed, MakeMutableSpan(reconstruced)));
  return distance_override->GetDistance(MakeDatapointPtr(original),
                                        MakeDatapointPtr(reconstruced));
}

absl::StatusOr<float> FixedPointIndexer::DistanceBetweenHashed(
    absl::Span<const uint8_t> hashed_1, absl::Span<const uint8_t> hashed_2,
    std::shared_ptr<const DistanceMeasure> distance_override) const {
  SCANN_RET_CHECK_EQ(hashed_1.size(), hashed_space_bytes());
  SCANN_RET_CHECK_EQ(hashed_2.size(), hashed_space_bytes());
  if (using_per_dimension_multipliers_ && per_dimension_bits_ == 8 &&
      scale_encoding_ == UNSPECIFIED_SCALE_ENCODING) {
    if (distance_override->specially_optimized_distance_tag() ==
        DistanceMeasure::DOT_PRODUCT) {
      return -DenseDotProduct(
          MakeDatapointPtr(reinterpret_cast<const int8_t*>(hashed_1.data()),
                           hashed_1.size()),
          MakeDatapointPtr(reinterpret_cast<const int8_t*>(hashed_2.data()),
                           hashed_2.size()),
          MakeDatapointPtr(sq_inv_multipliers_));
    } else if (distance_override->specially_optimized_distance_tag() ==
               DistanceMeasure::COSINE) {
      return 1.0 - DenseDotProduct(
                       MakeDatapointPtr(
                           reinterpret_cast<const int8_t*>(hashed_1.data()),
                           hashed_1.size()),
                       MakeDatapointPtr(
                           reinterpret_cast<const int8_t*>(hashed_2.data()),
                           hashed_2.size()),
                       MakeDatapointPtr(sq_inv_multipliers_));
    }
  }

  std::vector<float> reconstructed_1(original_space_dimension());
  std::vector<float> reconstructed_2(original_space_dimension());
  SCANN_RETURN_IF_ERROR(
      Reconstruct(hashed_1, MakeMutableSpan(reconstructed_1)));
  SCANN_RETURN_IF_ERROR(
      Reconstruct(hashed_2, MakeMutableSpan(reconstructed_2)));
  return distance_override->GetDistance(MakeDatapointPtr(reconstructed_1),
                                        MakeDatapointPtr(reconstructed_2));
}

absl::Status FixedPointIndexer::Reconstruct(
    const DatapointPtr<uint8_t>& encoded_dp,
    Datapoint<float>* reconstructed_dp) const {
  if (!encoded_dp.IsDense()) {
    return InvalidArgumentError(
        "FixedPointIndexer::Hash may only be called with dense "
        "datapoints.");
  }
  reconstructed_dp->mutable_values()->clear();
  reconstructed_dp->mutable_values()->resize(original_space_dimension());
  return Reconstruct(encoded_dp.values_span(),
                     reconstructed_dp->mutable_values_span());
}

absl::Status FixedPointIndexer::Reconstruct(
    absl::string_view encoded_str, Datapoint<float>* reconstructed_dp) const {
  reconstructed_dp->mutable_values()->clear();
  reconstructed_dp->mutable_values()->resize(original_space_dimension());
  auto span = MakeConstSpan(
      reinterpret_cast<const uint8_t*>(encoded_str.data()), encoded_str.size());
  return Reconstruct(span, reconstructed_dp->mutable_values_span());
}

}  // namespace research_scann
