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



#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_H_

#include <cstddef>
#include <cstdint>
#include <ostream>

#include "scann/utils/common.h"
#include "scann/utils/intrinsics/simd.h"
namespace research_scann {

#define SCANN_DISPATCH_INT8_TILE_HELPER(_namespace, _impl) \
  {                                                        \
    namespace simd_namespace = _namespace;                 \
    if (simd_namespace::RuntimeSupportsSimd()) {           \
      _impl;                                               \
    }                                                      \
  }
#define SCANN_DISPATCH_INT8_TILE_LAST_HELPER(_namespace, _impl) \
  {                                                             \
    namespace simd_namespace = _namespace;                      \
    _impl;                                                      \
  }

#ifdef __x86_64__

#ifdef SCANN_HAVE_AMX
#define SCANN_DISPATCH_INT8_TILE_AMX(_impl) \
  SCANN_DISPATCH_INT8_TILE_HELPER(amx, _impl)
#else
#define SCANN_DISPATCH_INT8_TILE_AMX(_impl)
#endif

#define SCANN_DISPATCH_INT8_TILE(_impl)               \
  SCANN_DISPATCH_INT8_TILE_AMX(_impl)                 \
  SCANN_DISPATCH_INT8_TILE_HELPER(avx512_vnni, _impl) \
  SCANN_DISPATCH_INT8_TILE_HELPER(avx512, _impl)      \
  SCANN_DISPATCH_INT8_TILE_HELPER(avx2, _impl)        \
  SCANN_DISPATCH_INT8_TILE_LAST_HELPER(fallback, _impl)

#elif HWY_HAVE_CONSTEXPR_LANES

#define SCANN_DISPATCH_INT8_TILE(_impl) \
  SCANN_DISPATCH_INT8_TILE_LAST_HELPER(highway, _impl)

#else

#define SCANN_DISPATCH_INT8_TILE(_impl) \
  SCANN_DISPATCH_INT8_TILE_LAST_HELPER(fallback, _impl)

#endif

enum class Int8TileSide {
  kQuery,
  kDatabase,
};
inline std::ostream& operator<<(std::ostream& os, Int8TileSide side) {
  switch (side) {
    case Int8TileSide::kQuery:
      return os << "kQuery";
    case Int8TileSide::kDatabase:
      return os << "kDatabase";
  }
}

class Int8TileCodec {
 public:
  virtual ~Int8TileCodec() = default;

  virtual void EncodeDatapoint(ConstSpan<int8_t> datapoint, size_t dp_idx,
                               MutableSpan<int8_t> payload) const = 0;
  virtual void ReconstructDatapoint(size_t dp_idx, ConstSpan<int8_t> payload,
                                    MutableSpan<int8_t> encoded) const = 0;
  virtual void ReconstructFloatDatapoint(size_t dp_idx,
                                         ConstSpan<int8_t> payload, float scale,
                                         MutableSpan<float> encoded) const = 0;

  size_t dimensionality() const { return dimensionality_; }
  size_t datapoint_bytes() const { return datapoint_bytes_; }
  size_t block_datapoints() const { return block_datapoints_; }
  size_t block_bytes() const { return block_bytes_; }
  size_t register_bytes() const { return register_bytes_; }
  PlatformGeneration platform_generation() const {
    return platform_generation_;
  }
  Int8TileSide side() const { return side_; }

 protected:
  Int8TileCodec(uint32_t dimensionality, size_t datapoint_bytes,
                size_t block_datapoints, size_t register_bytes,
                PlatformGeneration platform_generation, Int8TileSide side)
      : dimensionality_(dimensionality),
        datapoint_bytes_(datapoint_bytes),
        block_datapoints_(block_datapoints),
        block_bytes_(datapoint_bytes * block_datapoints),
        register_bytes_(register_bytes),
        platform_generation_(platform_generation),
        side_(side) {}

  const size_t dimensionality_;

  const size_t datapoint_bytes_;

  const size_t block_datapoints_;

  const size_t block_bytes_;
  const size_t register_bytes_;
  const PlatformGeneration platform_generation_;
  const Int8TileSide side_;
};
inline std::ostream& operator<<(std::ostream& os, const Int8TileCodec& codec) {
  return os << "Int8TileCodec{dimensionality_=" << codec.dimensionality()
            << ", datapoint_bytes_=" << codec.datapoint_bytes()
            << ", block_datapoints_=" << codec.block_datapoints()
            << ", block_bytes_=" << codec.block_bytes()
            << ", register_bytes_=" << codec.register_bytes()
            << ", platform_generation_=" << codec.platform_generation()
            << ", side_=" << codec.side() << "}";
}

template <typename Int8Tile>
constexpr size_t Int8TileBytes() {
  return Int8Tile::kPoints * Int8Tile::kDims;
}

class NoopInt8QueryExpander {
 public:
  void Ensure(size_t size) {}

  using ExpandedQueryT = int8_t;
  inline const ExpandedQueryT* Expand(ConstSpan<int8_t> queries) const {
    return queries.data();
  }
};

namespace amx {
class Int8DatabaseTile;
}
namespace avx512_vnni {
class Int8DatabaseTile;
}

#ifdef __x86_64__

namespace amx {
#define SCANN_SIMD_ATTRIBUTE SCANN_AMX

class Int8TileImpl {
 public:
  static constexpr size_t kPoints = 16;
  static constexpr size_t kDims = 64;

  static_assert(kPoints == Simd<int32_t>::kElementsPerRegister);
  static_assert(kDims == Simd<int8_t>::kElementsPerRegister);

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    __tile_loadd(&data_, ptr, kDims);
  }

 private:
  friend class Int32AccumulatorTile;
  __tile1024i data_ = {kPoints, kDims};
};

using Int8QueryExpander = NoopInt8QueryExpander;
class Int8QueryTile : public Int8TileImpl {};
class Int8DatabaseTile : public Int8TileImpl {};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() { __tile_zero(&data_); }

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    __tile_dpbssd(&data_, query.data_, database.data_);
  }

  using ResultType = Simd<int32_t, Int8QueryTile::kPoints>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    static_assert(alignof(result) >= 64);
    __tile_stored(&result, Int8QueryTile::kDims, data_);
  }

 private:
  __tile1024i data_ = {Int8QueryTile::kPoints, Int8QueryTile::kDims};
};

#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace amx

namespace avx512_vnni {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512_VNNI

using Int8QueryExpander = NoopInt8QueryExpander;
class Int8QueryTile {
 public:
  static constexpr size_t kPoints = 1;
  static constexpr size_t kDims = 4;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<int8_t>(
        Simd<int32_t>::Broadcast(*reinterpret_cast<const int32_t*>(ptr)));
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int8_t> data_;
};

class Int8DatabaseTile {
 public:
  static constexpr size_t kPoints = Simd<int32_t>::kNumElements;
  static constexpr size_t kDims = 4;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<uint8_t>::Load(reinterpret_cast<const uint8_t*>(ptr));
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<uint8_t> data_;
};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() = default;

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    data_ = _mm512_dpbusd_epi32(*data_, *database.data_, *query.data_);
  }

  using ResultType = Simd<int32_t>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    result = data_ + Simd<int32_t>::Broadcast(-128 * query_sums[0]);
  }

 private:
  Simd<int32_t> data_{Zeros()};
};

#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512_vnni

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512
#include "scann/distance_measures/many_to_many/int8_tile.inc"
#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/distance_measures/many_to_many/int8_tile.inc"
#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE SCANN_SSE4
#include "scann/distance_measures/many_to_many/int8_tile.inc"
#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4
#elif HWY_HAVE_CONSTEXPR_LANES

namespace highway {

std::unique_ptr<Int8TileCodec> NewInt8TileCodec(size_t dims, Int8TileSide side);

}

#endif

namespace fallback {
#define SCANN_SIMD_ATTRIBUTE

class Int8TileImpl {
 public:
  static constexpr size_t kPoints = 1;
  static constexpr size_t kDims = 1;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) { data_ = ptr[0]; }

 private:
  friend class Int32AccumulatorTile;
  int8_t data_;
};

using Int8QueryExpander = NoopInt8QueryExpander;
class Int8QueryTile : public Int8TileImpl {};
class Int8DatabaseTile : public Int8TileImpl {};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() = default;

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    data_ += static_cast<int32_t>(query.data_) * database.data_;
  }

  using ResultType = Simd<int32_t>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    result = data_;
  }

 private:
  int32_t data_ = 0;
};

#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace fallback
}  // namespace research_scann

#endif

#if defined(SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_TOGGLE
#undef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_TOGGLE
#else
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_INT8_TILE_TOGGLE
#endif

#include "scann/utils/intrinsics/highway.h"

#if !defined(__x86_64__) && HWY_HAVE_CONSTEXPR_LANES

HWY_BEFORE_NAMESPACE();
namespace research_scann {
namespace HWY_NAMESPACE {
#define SCANN_SIMD_ATTRIBUTE

using Int8QueryExpander = NoopInt8QueryExpander;

#if defined(__ARM_FEATURE_MATMUL_INT8) && HWY_TARGET_IS_NEON

using Int8QueryExpander = NoopInt8QueryExpander;

class Int8QueryTile {
 public:
  static constexpr size_t kPoints = 2;
  static constexpr size_t kDims = 8;

  static_assert(kPoints * kDims == Simd<int8_t>::kElementsPerRegister);

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<int8_t>::Load(ptr);
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int8_t> data_;
};

class Int8DatabaseTile {
 public:
  static constexpr size_t kPoints = 4;
  static constexpr size_t kDims = 8;

  static_assert(kPoints == Simd<int32_t>::kElementsPerRegister);

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<int8_t, 2>::Load(ptr);
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int8_t, 2> data_;
};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() = default;

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    for (int i : Seq(2)) {
      data_[i] = Simd<int32_t>::HwyType(vmmlaq_s32(
          (*data_[i]).raw, (*query.data_).raw, (*database.data_[i]).raw));
    }
  }

  using ResultType = Simd<int32_t, Int8QueryTile::kPoints>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    const hn::ScalableTag<int64_t> d64;
    const hn::ScalableTag<int32_t> d32;
    auto d0 = hn::BitCast(d64, *data_[0]);
    auto d1 = hn::BitCast(d64, *data_[1]);
    result[0] = hn::BitCast(d32, hn::InterleaveLower(d64, d0, d1));
    result[1] = hn::BitCast(d32, hn::InterleaveUpper(d64, d0, d1));
  }

 private:
  Simd<int32_t, 2> data_{Zeros()};
};

#else

class Int8QueryTile {
 public:
  static constexpr size_t kPoints = 1;
  static constexpr size_t kDims = 4;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<int8_t>(
        Simd<int32_t>::Broadcast(*reinterpret_cast<const int32_t*>(ptr)));
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int8_t> data_;
};

class Int8DatabaseTile {
 public:
  static constexpr size_t kPoints = Simd<int32_t>::kNumElements;
  static constexpr size_t kDims = 4;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) {
    data_ = Simd<int8_t>::Load(ptr);
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int8_t> data_;
};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() = default;

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    namespace hn = hwy::HWY_NAMESPACE;
    data_ = hn::SumOfMulQuadAccumulate(hn::ScalableTag<int32_t>(),
                                       *database.data_, *query.data_, *data_);
  }

  using ResultType = Simd<int32_t>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    result = data_;
  }

 private:
  Simd<int32_t> data_{Zeros()};
};
#endif

#include "scann/distance_measures/many_to_many/int8_tile_codec.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace HWY_NAMESPACE
}  // namespace research_scann

HWY_AFTER_NAMESPACE();

#endif
#endif
