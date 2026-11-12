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



#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "scann/distance_measures/many_to_many/many_to_many_sfp8.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "scann/distance_measures/many_to_many/int8_tile.h"
#include "scann/distance_measures/many_to_many/many_to_many_internal.h"
#include "scann/distance_measures/many_to_many/many_to_many_templates.h"

#if HWY_ONCE
namespace research_scann {

#if !defined(__x86_64__) && HWY_HAVE_CONSTEXPR_LANES
namespace {

template <typename CallbackT>
void DenseManyToManySFP8PretransposedImplDispatch(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(
      DenseManyToManySFP8PretransposedImpl<CallbackT>)
  (dist, queries, database, pool, std::move(callback));
}

template <typename CallbackT>
Status DenseManyToManySFP8OrthogonalityAmplifiedImplDispatch(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  HWY_EXPORT_T(Table, DenseManyToManySFP8OrthogonalityAmplifiedImpl<CallbackT>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(queries, normalized_residuals, lambda,
                                       database, pool, std::move(callback));
}

}  // namespace

namespace highway {

template <typename CallbackT>
void DenseManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  DenseManyToManySFP8PretransposedImplDispatch(dist, queries, database, pool,
                                               std::move(callback));
}

template <typename CallbackT>
Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  return DenseManyToManySFP8OrthogonalityAmplifiedImplDispatch(
      queries, normalized_residuals, lambda, database, pool,
      std::move(callback));
}

}  // namespace highway
#endif

namespace mm_internal {

template Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback);

template Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback);

template Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringOffsetWrapper<float> callback);

template Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback);

template Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback);

template Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringOffsetWrapper<float> callback);

}  // namespace mm_internal
}  // namespace research_scann
#endif
