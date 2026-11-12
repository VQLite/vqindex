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

#ifndef SCANN_PROJECTION_INTERNAL_RANDOM_ORTHOGONAL_MATRIX_CREATOR_H_
#define SCANN_PROJECTION_INTERNAL_RANDOM_ORTHOGONAL_MATRIX_CREATOR_H_

#include <cstdint>
#include <memory>

#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_random.h"

namespace research_scann {

unique_ptr<DenseDataset<float>> CreateRandomOrthogonalMatrix(
    int32_t input_dims, int32_t projected_dims, MTRandom& random);

}

#endif
