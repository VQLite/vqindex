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

#include "scann/hashes/coder/fixed_point_coder.h"

#include <memory>
#include <vector>

#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/utils/scalar_quantization_helpers.h"

namespace research_scann {
namespace one_to_many_low_level {

template <typename DatasetView, typename CallbackFunctor>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatDispatch(
    const float* query, const DatasetView* __restrict__ database,
    MutableSpan<float> result, CallbackFunctor* __restrict__ callback) {
  DenseDotProductDistanceOneToManyInt8FloatLowLevel<DatasetView, false,
                                                    uint32_t>(
      query, database, nullptr, result, callback);
}

}  // namespace one_to_many_low_level

absl::StatusOr<float> FixedPointCoder::DistanceBetweenHashed(
    absl::Span<const uint8_t> hashed_1,
    absl::Span<const uint8_t> hashed_2) const {
  if (!symmetric_distance_) {
    return absl::FailedPreconditionError(
        "Must call SetSymmetricDistanceMeasure once before computing symmetric "
        "distance");
  }
  return indexer_.DistanceBetweenHashed(hashed_1, hashed_2,
                                        symmetric_distance_);
}

template <bool kIsCosine>
absl::StatusOr<NNResultsVector>
FixedPointCoder::FixedPointAsymmetricScoreDotOrCosine(
    absl::Span<const float> query,
    const DenseDatasetView<int8_t>* __restrict__ dataset_ptr,
    Coder::ScoringOption option) const {
  std::vector<float> results(dataset_ptr->size());
  std::unique_ptr<float[]> pre_processed_query;
  if (indexer_.using_per_dimension_multipliers_) {
    pre_processed_query = PrepareForAsymmetricScalarQuantizedDotProduct(
        MakeDatapointPtr(query), indexer_.inv_multipliers_);
  } else {
    std::vector<float> inv_multipliers(original_space_dimension(),
                                       indexer_.universal_inv_multiplier_);
    pre_processed_query = PrepareForAsymmetricScalarQuantizedDotProduct(
        MakeDatapointPtr(query), inv_multipliers);
  }
  auto results_span = MakeMutableSpan(results);
  auto set_distance_functor =
      one_to_many_low_level::SetDistanceFunctor<float>(results_span);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      pre_processed_query.get(), dataset_ptr, results_span,
      &set_distance_functor);
  int num_datapoints = dataset_ptr->size();
  bool compute_score_only_without_sorting =
      option.num_neighbors >= num_datapoints && !option.sort_results;
  if (compute_score_only_without_sorting) {
    NNResultsVector res;
    res.resize(num_datapoints);

    for (size_t i : Seq(dataset_ptr->size())) {
      float d = results[i];
      if (kIsCosine) {
        d += 1;
      }
      res[i].first = i;
      res[i].second = d;
    }
    return res;
  } else {
    TopNeighbors<float> top_n(option.num_neighbors);
    for (size_t i : Seq(dataset_ptr->size())) {
      float d = results[i];
      if (kIsCosine) {
        d += 1;
      }
      if (d < option.epsilon) {
        top_n.push(i, d);
      }
    }
    return option.sort_results ? top_n.Take() : top_n.TakeUnsorted();
  }
}

absl::StatusOr<NNResultsVector> FixedPointCoder::AsymmetricScore(
    absl::Span<const float> query,
    std::shared_ptr<DenseDatasetView<uint8_t>> dataset,
    const Coder::ScoringOption& option) const {
  std::shared_ptr<DistanceMeasure> dist =
      option.scoring_distance == nullptr
          ? std::make_shared<DotProductDistance>()
          : option.scoring_distance;
  if (indexer_.scale_encoding() == UNSPECIFIED_SCALE_ENCODING &&
      indexer_.per_dimension_bits() == 8) {
    switch (dist->specially_optimized_distance_tag()) {
      case DistanceMeasure::DOT_PRODUCT:
        return FixedPointAsymmetricScoreDotOrCosine<false>(
            query, reinterpret_cast<DenseDatasetView<int8_t>*>(dataset.get()),
            option);
      case DistanceMeasure::COSINE:
        return FixedPointAsymmetricScoreDotOrCosine<true>(
            query, reinterpret_cast<DenseDatasetView<int8_t>*>(dataset.get()),
            option);
      default:
        break;
    }
  }

  bool compute_score_only_without_sorting =
      option.num_neighbors >= dataset->size() && !option.sort_results;
  if (compute_score_only_without_sorting) {
    NNResultsVector res;
    int num_datapoints = dataset->size();
    res.resize(num_datapoints);
    size_t dim = dataset->dimensionality();
    for (size_t i : Seq(num_datapoints)) {
      SCANN_ASSIGN_OR_RETURN(
          float d, indexer_.DistanceBetweenOriginalAndHashed(
                       query, MakeConstSpan(dataset->GetPtr(i), dim), dist));
      res[i].first = i;
      res[i].second = d;
    }
    return res;
  } else {
    TopNeighbors<float> top_n(option.num_neighbors);
    size_t dim = dataset->dimensionality();
    for (size_t i : Seq(dataset->size())) {
      SCANN_ASSIGN_OR_RETURN(
          float d, indexer_.DistanceBetweenOriginalAndHashed(
                       query, MakeConstSpan(dataset->GetPtr(i), dim), dist));
      top_n.push(i, d);
    }
    return option.sort_results ? top_n.Take() : top_n.TakeUnsorted();
  }
}

}  // namespace research_scann
