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

#ifndef SCANN_PROJECTION_BLOCKWISE_RANDOM_ORTHOGONAL_H_
#define SCANN_PROJECTION_BLOCKWISE_RANDOM_ORTHOGONAL_H_

#include <cstdint>
#include <memory>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/projection/projection_base.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class BlockwiseRandomOrthogonalProjection : public Projection<T> {
 public:
  BlockwiseRandomOrthogonalProjection(int32_t input_dims, int32_t block_size,
                                      int32_t seed);

  void Create();

  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<float>* projected) const final;
  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<double>* projected) const final;

 private:
  template <typename FloatT>
  Status ProjectInputImpl(const DatapointPtr<T>& input,
                          Datapoint<FloatT>* projected) const;

  void MakePermutation();

  template <typename FloatT>
  void ApplyPermutation(ConstSpan<FloatT> input,
                        MutableSpan<FloatT> output) const;

  int32_t input_dims_;
  int32_t block_size_;

  vector<uint32_t> permutation_;
  unique_ptr<MTRandom> random_;
  int32_t seed_;
  unique_ptr<const DenseDataset<float>> block_rotation_matrix_;
  unique_ptr<const DenseDataset<float>> remainder_matrix_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, BlockwiseRandomOrthogonalProjection);

}  // namespace research_scann

#endif
