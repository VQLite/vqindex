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

#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_INTERNAL_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_INTERNAL_H_

#include <cstddef>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/many_to_many/fp8_transposed.h"
#include "scann/distance_measures/many_to_many/many_to_many_templates.h"
#include "scann/distance_measures/many_to_many/sfp8_transposed.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace mm_internal {

inline bool IsSupportedDistanceMeasure(const DistanceMeasure& dist) {
  switch (dist.specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
    case DistanceMeasure::SQUARED_L2:
    case DistanceMeasure::COSINE:
      return true;
    default:
      return false;
  }
}

template <typename FloatT, typename CallbackT>
void CallOneToManyDistance(const DistanceMeasure& dist,
                           DefaultDenseDatasetView<FloatT> queries,
                           const DefaultDenseDatasetView<FloatT>& database,
                           ThreadPool* pool, CallbackT callback) {
  auto one_query_results_storage = std::make_unique<FloatT[]>(database.size());
  MutableSpan<FloatT> one_query_results(one_query_results_storage.get(),
                                        database.size());
  const size_t query_dims = queries.dimensionality();
  for (size_t query_idx : IndicesOf(queries)) {
    DatapointPtr<FloatT> q(nullptr, queries.GetPtr(query_idx), query_dims,
                           query_dims);
    DenseDistanceOneToMany(dist, q, &database, one_query_results, pool);
    callback(one_query_results, 0, query_idx);
  }
}

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl2(
    const DistanceMeasure& dist, DefaultDenseDatasetView<FloatT> queries,
    const DefaultDenseDatasetView<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");
  DCHECK_GE(queries.size(), 2);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                               callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else {
    return avx1::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  }

#else
  return HWY_NAMESPACE::DenseDistanceManyToManyImpl(dist, queries, database,
                                                    pool, std::move(callback));
#endif
}

template <typename DatabaseT, typename CallbackT>
void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DefaultDenseDatasetView<float>& queries,
    const DefaultDenseDatasetView<float>& normalized_residuals, float lambda,
    const DatabaseT& database, ThreadPool* pool, CallbackT callback) {
#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseManyToManyOrthogonalityAmplifiedImpl(
        queries, normalized_residuals, lambda, database, pool,
        std::move(callback));
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseManyToManyOrthogonalityAmplifiedImpl(
        queries, normalized_residuals, lambda, database, pool,
        std::move(callback));
  } else {
    return avx1::DenseManyToManyOrthogonalityAmplifiedImpl(
        queries, normalized_residuals, lambda, database, pool,
        std::move(callback));
  }

#else
  return HWY_NAMESPACE::DenseManyToManyOrthogonalityAmplifiedImpl(
      queries, normalized_residuals, lambda, database, pool,
      std::move(callback));
#endif
}

template <typename CallbackT>
Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  SCANN_DISPATCH_INT8_TILE(
      return simd_namespace::DenseManyToManySFP8OrthogonalityAmplifiedImpl(
          queries, normalized_residuals, lambda, database, pool,
          std::move(callback)));
}

template <typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyFP8PretransposedImpl2(
    const DistanceMeasure& dist, const DefaultDenseDatasetView<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  DCHECK_GE(queries.size(), 1);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                       pool, callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else if (RuntimeSupportsAvx1()) {
    return avx1::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  }

#else
  return HWY_NAMESPACE::DenseManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
#endif
}

template <typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManySFP8PretransposedImpl2(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  DCHECK_GE(queries.size(), 1);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);
  DCHECK_EQ(queries.dimensionality(), database.dimensionality());
  DCHECK_EQ(queries.platform_generation(), database.platform_generation());

  SCANN_DISPATCH_INT8_TILE(
      return simd_namespace::DenseManyToManySFP8PretransposedImpl(
          dist, queries, database, pool, std::move(callback)));
}

template <typename FloatT, typename CallbackT>
void DenseDistanceManyToManyImpl(
    const DistanceMeasure& dist, DefaultDenseDatasetView<FloatT> queries,
    const DefaultDenseDatasetView<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");

  if (database.empty() || queries.size() == 0) return;

  if (queries.size() == 1 || !IsSupportedDistanceMeasure(dist)) {
    return CallOneToManyDistance(dist, queries, database, pool,
                                 std::move(callback));
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper =
        [&callback](MutableSpan<FloatT> block_distances,
                    DatapointIndex base_dp_idx, DatapointIndex query_idx) {
          for (auto& elem : block_distances) {
            elem += static_cast<FloatT>(1.0);
          }
          callback(block_distances, base_dp_idx, query_idx);
        };
    return DenseDistanceManyToManyImpl2<FloatT>(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    return DenseDistanceManyToManyImpl2<FloatT, CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
}

template <typename CallbackT>
Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  if (queries.empty()) return OkStatus();
  if (database.empty()) return OkStatus();

  if (!IsSupportedDistanceMeasure(dist)) {
    return InvalidArgumentError(
        "DenseDistanceManyToManySFP8Pretransposed only supports dot product, "
        "cosine and squared L2 distance.");
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper = [&callback](MutableSpan<float> block_distances,
                                             DatapointIndex base_dp_idx,
                                             DatapointIndex query_idx) {
      for (auto& elem : block_distances) {
        elem += static_cast<float>(1.0);
      }
      callback(block_distances, base_dp_idx, query_idx);
    };
    DenseDistanceManyToManySFP8PretransposedImpl2(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    DenseDistanceManyToManySFP8PretransposedImpl2<CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
  return OkStatus();
}

template <typename CallbackT>
Status DenseDistanceManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DefaultDenseDatasetView<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  if (queries.empty()) return OkStatus();

  if (!IsSupportedDistanceMeasure(dist)) {
    return InvalidArgumentError(
        "DenseDistanceManyToManyFP8Pretransposed only supports dot product, "
        "cosine and squared L2 distance.");
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper = [&callback](MutableSpan<float> block_distances,
                                             DatapointIndex base_dp_idx,
                                             DatapointIndex query_idx) {
      for (auto& elem : block_distances) {
        elem += static_cast<float>(1.0);
      }
      callback(block_distances, base_dp_idx, query_idx);
    };
    DenseDistanceManyToManyFP8PretransposedImpl2(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    DenseDistanceManyToManyFP8PretransposedImpl2<CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
  return OkStatus();
}

}  // namespace mm_internal
}  // namespace research_scann

#endif
