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

#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_

#include <cstddef>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/many_to_many/fp8_transposed.h"
#include "scann/distance_measures/many_to_many/many_to_many_common.h"
#include "scann/distance_measures/many_to_many/many_to_many_flags.h"
#include "scann/distance_measures/many_to_many/sfp8_transposed.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/fma.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"

#define SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, kBatchSize, function, ...) \
  case kBatchSize:                                                         \
    if constexpr (kBatchSize <= kMaxBatchSize) {                           \
      function<kBatchSize>(__VA_ARGS__);                                   \
      break;                                                               \
    } else {                                                               \
      ABSL_FALLTHROUGH_INTENDED;                                           \
    }

#define SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kMaxBatchSize, batch_size,   \
                                             function, ...)               \
  static_assert(kMaxBatchSize <= 32, "Max batch size must be <= 32");     \
  switch (batch_size) {                                                   \
    case 0:                                                               \
      break;                                                              \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 1, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 2, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 3, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 4, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 5, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 6, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 7, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 8, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 9, function, __VA_ARGS__);  \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 10, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 11, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 12, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 13, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 14, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 15, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 16, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 17, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 18, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 19, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 20, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 21, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 22, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 23, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 24, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 25, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 26, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 27, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 28, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 29, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 30, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 31, function, __VA_ARGS__); \
      SCANN_MM_BATCH_SIZE_CASE(kMaxBatchSize, 32, function, __VA_ARGS__); \
    default:                                                              \
      DLOG(FATAL) << "Invalid Batch Size:  " << batch_size;               \
  }

namespace research_scann {

inline float SFP8DotProduct(size_t dims, ConstSpan<uint8_t> a, float a_scale,
                            ConstSpan<uint8_t> b, float b_scale) {
  const int8_t* signed_a = reinterpret_cast<const int8_t*>(a.data());
  const int8_t* signed_b = reinterpret_cast<const int8_t*>(b.data());
  int32_t sum = 0;

  for (size_t i = 0; i < dims; ++i) {
    sum += signed_a[i] * signed_b[i];
  }
  return sum * (a_scale * b_scale);
}

#ifdef __x86_64__

namespace avx1 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#include "scann/distance_measures/many_to_many/many_to_many_sfp8_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#include "scann/distance_measures/many_to_many/many_to_many_sfp8_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

namespace avx512_vnni {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512_VNNI
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#include "scann/distance_measures/many_to_many/many_to_many_sfp8_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512_vnni

namespace amx {
#define SCANN_SIMD_ATTRIBUTE SCANN_AMX
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#include "scann/distance_measures/many_to_many/many_to_many_sfp8_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace amx

#elif HWY_HAVE_CONSTEXPR_LANES

namespace highway {

template <typename CallbackT>
void DenseManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT);

template <typename CallbackT>
Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT);

}  // namespace highway

#endif

namespace fallback {

SCANN_INLINE void DenseManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback) {
  std::vector<uint8_t> query(queries.hashed_space_bytes());
  std::vector<uint8_t> dp(database.hashed_space_bytes());

  for (size_t i : IndicesOf(queries)) {
    CHECK_OK(queries.ReconstructDatapoint(i, MakeMutableSpan(query)));
    const float query_scale = queries.scales()[i];
    for (size_t j : IndicesOf(database)) {
      CHECK_OK(database.ReconstructDatapoint(j, MakeMutableSpan(dp)));
      const float dp_scale = database.scales()[j];
      float result = -SFP8DotProduct(queries.dimensionality(), query,
                                     query_scale, dp, dp_scale);
      if (dist.specially_optimized_distance_tag() ==
          DistanceMeasure::SQUARED_L2) {
        result = queries.squared_l2_norms()[i] +
                 database.squared_l2_norms()[j] + 2 * result;
      }
      callback(MakeMutableSpan(&result, 1), j, i);
    }
  }
}

SCANN_INLINE Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback) {
  const size_t dims = database.dimensionality();

  std::vector<uint8_t> q(queries.hashed_space_bytes());
  std::vector<uint8_t> r(normalized_residuals.hashed_space_bytes());
  std::vector<uint8_t> dp(database.hashed_space_bytes());

  std::vector<float> r_dot_qs(queries.size());
  for (const size_t i : IndicesOf(queries)) {
    SCANN_RETURN_IF_ERROR(
        normalized_residuals.ReconstructDatapoint(i, MakeMutableSpan(r)));
    const float r_scale = normalized_residuals.scales()[i];
    SCANN_RETURN_IF_ERROR(queries.ReconstructDatapoint(i, MakeMutableSpan(q)));
    const float q_scale = queries.scales()[i];
    r_dot_qs[i] = SFP8DotProduct(dims, r, r_scale, q, q_scale);
  }
  auto callback_wrapper = [&](MutableSpan<float> block_distances,
                              DatapointIndex base_dp_idx,
                              DatapointIndex query_idx) {
    CHECK_OK(normalized_residuals.ReconstructDatapoint(query_idx,
                                                       MakeMutableSpan(r)));
    const float r_scale = normalized_residuals.scales()[query_idx];
    const float r_dot_q = r_dot_qs[query_idx];
    for (size_t j : IndicesOf(block_distances)) {
      CHECK_OK(
          database.ReconstructDatapoint(base_dp_idx + j, MakeMutableSpan(dp)));
      const float dp_scale = database.scales()[base_dp_idx + j];
      const float r_dot_d = SFP8DotProduct(dims, r, r_scale, dp, dp_scale);
      block_distances[j] += lambda * (r_dot_q * r_dot_q + r_dot_d * r_dot_d -
                                      2 * r_dot_q * r_dot_d);
    }
    callback(block_distances, base_dp_idx, query_idx);
  };

  DenseManyToManySFP8PretransposedImpl(SquaredL2Distance(), queries, database,
                                       nullptr, std::move(callback_wrapper));
  return OkStatus();
}

}  // namespace fallback
}  // namespace research_scann

#endif

#if defined(SCANN_DISTANCE_MEASURES_MANY_TO_MANY_TEMPLATES_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_TEMPLATES_TOGGLE
#undef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_TEMPLATES_TOGGLE
#else
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_TEMPLATES_TOGGLE
#endif

#include "scann/utils/intrinsics/highway.h"

#if !defined(__x86_64__) && HWY_HAVE_CONSTEXPR_LANES

HWY_BEFORE_NAMESPACE();
namespace research_scann {
namespace HWY_NAMESPACE {
#define SCANN_SIMD_ATTRIBUTE
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#include "scann/distance_measures/many_to_many/many_to_many_sfp8_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace HWY_NAMESPACE
}  // namespace research_scann
HWY_AFTER_NAMESPACE();

#endif
#endif
