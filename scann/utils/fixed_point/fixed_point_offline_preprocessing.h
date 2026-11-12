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



#ifndef SCANN_UTILS_FIXED_POINT_FIXED_POINT_OFFLINE_PREPROCESSING_H_
#define SCANN_UTILS_FIXED_POINT_FIXED_POINT_OFFLINE_PREPROCESSING_H_

#include <math.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/fixed_point_offline_preprocessing.h"
#include "scann/utils/fixed_point/fixed_point_training_results.pb.h"
#include "scann/utils/scale_encoding_helpers.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {

class FixedPointTrainer {
 public:
  virtual ~FixedPointTrainer();

  virtual absl::Status AddDatapoint(const DatapointPtr<float>& dptr) = 0;

  virtual FixedPointTrainingResults ExtractResults() = 0;

  static unique_ptr<FixedPointTrainer> New(float multiplier_quantile,
                                           DatapointIndex num_datapoints);
};

class FixedPointMaxTrainer : public FixedPointTrainer {
 public:
  absl::Status AddDatapoint(const DatapointPtr<float>& dptr) override;
  FixedPointTrainingResults ExtractResults() override;

 private:
  bool add_datapoint_called_at_least_once_ = false;

  FixedPointTrainingResults results_;

  bool results_extracted_ = false;
};

class FixedPointQuantileTrainer : public FixedPointTrainer {
 public:
  FixedPointQuantileTrainer(float multiplier_quantile,
                            DatapointIndex num_datapoints);

  absl::Status AddDatapoint(const DatapointPtr<float>& dptr) override;
  FixedPointTrainingResults ExtractResults() override;

 private:
  DatapointIndex num_datapoints_added_ = 0;

  DatapointIndex num_datapoints_ = 0;

  float multiplier_quantile_;

  vector<TopNAmortizedConstant<float>> top_ns_;
};

class FixedPointGfvConverter {
 public:
  enum Distance {
    ALL,

    DOT_PRODUCT,

    COSINE,

    SQUARED_L2,
  };

  explicit FixedPointGfvConverter(
      const FixedPointTrainingResults& training_results,
      Distance distance = ALL, double noise_shaping_threshold = NAN);

  explicit FixedPointGfvConverter(vector<float> per_dimension_multipliers,
                                  Distance distance = ALL);

  explicit FixedPointGfvConverter(float multiplier, Distance distance = ALL,
                                  double noise_shaping_threshold = NAN);

  absl::Status Convert(GenericFeatureVector* gfv) const;

 private:
  Distance distance_ = ALL;

  vector<float> multipliers_;

  bool using_per_dimension_multipliers_ = true;

  float universal_multiplier_ = NAN;

  double noise_shaping_threshold_ = NAN;
};

class FixedPointCoder;

class FixedPointIndexer {
 public:
  explicit FixedPointIndexer() {}

  FixedPointIndexer(ScaleEncoding scale_encoding, int per_dimension_bits,
                    const FixedPointTrainingResults& training_results)
      : FixedPointIndexer(scale_encoding, per_dimension_bits,
                          training_results.max_abs_value_by_dim_size()) {
    multipliers_.reserve(training_results.max_abs_value_by_dim_size());
    for (float f : training_results.max_abs_value_by_dim()) {
      float multiplier = numeric_limits<int8_t>::max() / f;
      float inv_multiplier = 1.0 / multiplier;
      multipliers_.push_back(multiplier);
      inv_multipliers_.push_back(inv_multiplier);
      sq_inv_multipliers_.push_back(inv_multiplier * inv_multiplier);
    }
  }

  FixedPointIndexer(ScaleEncoding scale_encoding, int per_dimension_bits,
                    vector<float> per_dimension_multipliers)
      : FixedPointIndexer(scale_encoding, per_dimension_bits,
                          per_dimension_multipliers.size()) {
    multipliers_ = std::move(per_dimension_multipliers);
    for (float f : multipliers_) {
      float inv_multiplier = 1.0 / f;
      inv_multipliers_.push_back(inv_multiplier);
      sq_inv_multipliers_.push_back(inv_multiplier * inv_multiplier);
    }
  }

  explicit FixedPointIndexer(vector<float> per_dimension_multipliers)
      : FixedPointIndexer(UNSPECIFIED_SCALE_ENCODING, 8,
                          std::move(per_dimension_multipliers)) {}

  FixedPointIndexer(ScaleEncoding scale_encoding, int per_dimension_bits,
                    float multiplier, DimensionIndex dimension)
      : FixedPointIndexer(scale_encoding, per_dimension_bits, dimension) {
    universal_multiplier_ = multiplier;
    universal_inv_multiplier_ = 1.0 / multiplier;
    using_per_dimension_multipliers_ = false;
  }

  DatapointIndex original_space_dimension() const { return dimension_; }

  DatapointIndex hashed_space_bytes() const { return hashed_bytes_; }

  ScaleEncoding scale_encoding() const { return scale_encoding_; }

  int per_dimension_bits() const { return per_dimension_bits_; }

  absl::Status Hash(const DatapointPtr<float>& original_dp,
                    Datapoint<uint8_t>* encoded_dp) const;

  absl::Status Hash(const DatapointPtr<float>& original_dp,
                    std::string* encoded_str) const;

  absl::Status Hash(ConstSpan<float> original,
                    MutableSpan<uint8_t> encoded) const;

  absl::Status HashWithNoiseShaping(ConstSpan<float> original,
                                    MutableSpan<uint8_t> encoded,
                                    const double noise_shaping_threshold) const;

  ABSL_DEPRECATED("Pass ScaleEncoding to the constructor instead.")
  absl::StatusOr<std::string> HashScaledWithNoiseShaping(
      int bits, absl::Span<const float> original, ScaleEncoding scale_encoding,
      double noise_shaping_threshold) const;

  absl::Status Reconstruct(const DatapointPtr<uint8_t>& encoded_dp,
                           Datapoint<float>* reconstructed_dp) const;

  absl::Status Reconstruct(absl::string_view encoded_str,
                           Datapoint<float>* reconstructed_dp) const;

  absl::Status Reconstruct(ConstSpan<uint8_t> encoded,
                           MutableSpan<float> reconstructed) const;

  absl::StatusOr<float> DistanceBetweenOriginalAndHashed(
      absl::Span<const float> original, absl::Span<const uint8_t> hashed,
      std::shared_ptr<const DistanceMeasure> distance_override) const;

  absl::StatusOr<float> DistanceBetweenHashed(
      absl::Span<const uint8_t> hashed_1, absl::Span<const uint8_t> hashed_2,
      std::shared_ptr<const DistanceMeasure> distance_override) const;

 private:
  FixedPointIndexer(ScaleEncoding scale_encoding, int per_dimension_bits,
                    DimensionIndex dimension)
      : scale_encoding_(scale_encoding),
        per_dimension_bits_(per_dimension_bits),
        dimension_(dimension),
        hashed_bytes_(*ScaledDatapointEncodedBytes(
            per_dimension_bits_, scale_encoding_, dimension_)) {}

  ScaleEncoding scale_encoding_;

  int per_dimension_bits_;

  DimensionIndex dimension_;

  DimensionIndex hashed_bytes_;

  vector<float> multipliers_;

  float universal_multiplier_ = NAN;

  vector<float> inv_multipliers_;

  vector<float> sq_inv_multipliers_;

  float universal_inv_multiplier_ = NAN;

  bool using_per_dimension_multipliers_ = true;

  friend class FixedPointCoder;
};

}  // namespace research_scann

#endif
