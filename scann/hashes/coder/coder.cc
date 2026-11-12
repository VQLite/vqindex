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

#include "scann/hashes/coder/coder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

absl::StatusOr<GenericFeatureVector> Coder::EncodeGFV(
    const GenericFeatureVector& original_gfv) const {
  SCANN_ASSIGN_OR_RETURN(auto tag, DetectInMemoryTypeFromGfv(original_gfv));
  if (tag != TagForType<float>()) {
    return absl::InvalidArgumentError("Coder/Codec only supports FLOAT GFVs");
  }
  if (!original_gfv.feature_index().empty()) {
    return absl::InvalidArgumentError("Coder/Codec only supports dense GFVs");
  }
  Datapoint<float> original_dp;
  SCANN_RETURN_IF_ERROR(original_dp.FromGfv(original_gfv));
  GenericFeatureVector encoded_gfv = original_gfv;
  encoded_gfv.clear_feature_value_float();
  encoded_gfv.set_feature_type(GenericFeatureVector::STRING);
  SCANN_RETURN_IF_ERROR(EncodeDatapoint(
      original_dp.ToPtr(), encoded_gfv.mutable_feature_value_string()));
  return encoded_gfv;
}

absl::StatusOr<GenericFeatureVector> Coder::DecodeGFV(
    const GenericFeatureVector& encoded_gfv) const {
  if (encoded_gfv.feature_type() != GenericFeatureVector::STRING) {
    return absl::InvalidArgumentError("EncodedGFV must use STRING type.");
  }
  if (!encoded_gfv.feature_index().empty()) {
    return absl::InvalidArgumentError("Coder/Codec only supports dense GFVs");
  }
  Datapoint<float> reconstructed_dp;
  SCANN_RETURN_IF_ERROR(
      DecodeDatapoint(encoded_gfv.feature_value_string(), &reconstructed_dp));
  GenericFeatureVector reconstructed_gfv = encoded_gfv;
  reconstructed_gfv.set_feature_type(GenericFeatureVector::FLOAT);
  reconstructed_gfv.clear_feature_value_string();
  reconstructed_gfv.mutable_feature_value_float()->Reserve(
      reconstructed_dp.dimensionality());
  for (float f : reconstructed_dp.values_span()) {
    reconstructed_gfv.add_feature_value_float(f);
  }
  return reconstructed_gfv;
}

absl::StatusOr<NNResultsVector> Coder::SymmetricScore(
    absl::Span<const uint8_t> query, const DenseDatasetView<uint8_t>& dataset,
    ScoringOption option) const {
  if (option.scoring_distance != nullptr) {
    return absl::InvalidArgumentError(
        "Setting symmetric distance measure should be done in "
        "SetSymmetricDistanceMeasure mehod instead of passing it through "
        "ScoringOption argument");
  }
  TopNeighbors<float> top_n(option.num_neighbors);
  size_t dim = dataset.dimensionality();
  for (auto i : Seq(dataset.size())) {
    SCANN_ASSIGN_OR_RETURN(float d,
                           this->DistanceBetweenHashed(
                               query, MakeConstSpan(dataset.GetPtr(i), dim)));
    top_n.push(i, d);
  }
  if (option.epsilon < std::numeric_limits<float>::infinity()) {
    auto result = top_n.TakeUnsorted();
    auto it =
        std::partition(result.begin(), result.end(),
                       [&option](const std::pair<DatapointIndex, float>& arg) {
                         return arg.second <= option.epsilon;
                       });
    result.resize(it - result.begin());
    if (option.sort_results) {
      ZipSortBranchOptimized(DistanceComparatorBranchOptimized(),
                             result.begin(), result.end());
    }
    return result;
  }
  return option.sort_results ? top_n.Take() : top_n.TakeUnsorted();
}

}  // namespace research_scann
