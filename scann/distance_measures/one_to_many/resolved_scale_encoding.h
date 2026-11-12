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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_RESOLVED_SCALE_ENCODING_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_RESOLVED_SCALE_ENCODING_H_

#include "absl/log/log.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/scalar_quantization_helpers.h"

namespace research_scann {

inline constexpr size_t kCachelineBytes = 64;

enum class ResolvedScaleEncoding {
  kNone = 0,
  kFloat32ScaleSuffix = 1,
  kFloat32ScaleBottomBits = 2,
};

inline ResolvedScaleEncoding ResolveScaleEncoding(int bits,
                                                  ScaleEncoding scale_encoding,
                                                  size_t dimension) {
  switch (scale_encoding) {
    case FLOAT32_SCALE_SUFFIX:
      return ResolvedScaleEncoding::kFloat32ScaleSuffix;

    case FLOAT32_SCALE_BOTTOM_BITS:
      return ResolvedScaleEncoding::kFloat32ScaleBottomBits;

    case FLOAT32_SCALE_AUTO_BOTTOM_BITS: {
      if (dimension < kMinDimensionsForBottomBits) {
        return ResolvedScaleEncoding::kFloat32ScaleSuffix;
      }
      const size_t bytes = (bits == 4) ? DivRoundUp(dimension, 2) : dimension;
      if (bytes % kCachelineBytes != 0) {
        return ResolvedScaleEncoding::kFloat32ScaleSuffix;
      }

      if (bytes > 6 * kCachelineBytes) {
        return ResolvedScaleEncoding::kFloat32ScaleSuffix;
      }
      return ResolvedScaleEncoding::kFloat32ScaleBottomBits;
    }

    case UNSPECIFIED_SCALE_ENCODING:
      return ResolvedScaleEncoding::kNone;
  }
  LOG(FATAL) << "Unknown scale encoding: " << scale_encoding;
}

}  // namespace research_scann
#endif
