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
#define HWY_TARGET_INCLUDE "scann/distance_measures/many_to_many/int8_tile.cc"
#include "scann/distance_measures/many_to_many/int8_tile.h"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#if HWY_ONCE
#if !defined(__x86_64__) && HWY_HAVE_CONSTEXPR_LANES

namespace research_scann {
namespace {
HWY_EXPORT(NewInt8TileCodec);
}

namespace highway {

std::unique_ptr<Int8TileCodec> NewInt8TileCodec(size_t dims,
                                                Int8TileSide side) {
  return HWY_DYNAMIC_DISPATCH(NewInt8TileCodec)(dims, side);
}

}  // namespace highway
}  // namespace research_scann
#endif
#endif
