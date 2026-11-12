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

#include "scann/projection/blockwise_random_orthogonal.h"

#include "scann/data_format/datapoint.h"
#include "scann/projection/internal/random_orthogonal_matrix_creator.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"

namespace research_scann {

template <typename T>
BlockwiseRandomOrthogonalProjection<T>::BlockwiseRandomOrthogonalProjection(
    int32_t input_dims, int32_t block_size, int32_t seed)
    : input_dims_(input_dims), block_size_(block_size), seed_(seed) {
  CHECK_GT(input_dims_, 0) << "Input dimensionality must be > 0";
  CHECK_GT(block_size_, 1) << "Block size must be > 1";
}

template <typename T>
void BlockwiseRandomOrthogonalProjection<T>::Create() {
  random_ = std::make_unique<MTRandom>(seed_);
  MakePermutation();

  if (input_dims_ >= block_size_) {
    block_rotation_matrix_ =
        CreateRandomOrthogonalMatrix(block_size_, block_size_, *random_);
  }
  if (input_dims_ % block_size_ > 0) {
    remainder_matrix_ = CreateRandomOrthogonalMatrix(
        input_dims_ % block_size_, input_dims_ % block_size_, *random_);
  }
}

template <typename T>
template <typename FloatT>
Status BlockwiseRandomOrthogonalProjection<T>::ProjectInputImpl(
    const DatapointPtr<T>& input, Datapoint<FloatT>* projected) const {
  CHECK(projected != nullptr);
  projected->clear();
  projected->mutable_values()->resize(input_dims_);

  if (!block_rotation_matrix_ && !remainder_matrix_) {
    return FailedPreconditionError(
        "Create the random orthogonal matrix first.");
  }

  const size_t num_complete_blocks = input_dims_ / block_size_;
  vector<FloatT> tmp_output(input_dims_);

  ConstSpan<uint32_t> permutation = MakeConstSpan(permutation_);
  auto do_block_round1 = [&](size_t block_start, size_t block_end,
                             const DenseDataset<float>& rotation_matrix) {
    const size_t current_block_size = block_end - block_start;
    DatapointPtr<T> block = MakeDatapointPtr(
        input.values_span().subspan(block_start, current_block_size));
    for (size_t offset_idx : Seq(current_block_size)) {
      const size_t global_dim_idx = block_start + offset_idx;
      const uint32_t permuted_dim_idx = permutation[global_dim_idx];
      tmp_output[permuted_dim_idx] =
          DotProduct(block, rotation_matrix[offset_idx]);
    }
  };

  for (size_t block_idx : Seq(num_complete_blocks)) {
    do_block_round1(block_idx * block_size_, (block_idx + 1) * block_size_,
                    *block_rotation_matrix_);
  }
  if (remainder_matrix_) {
    do_block_round1(num_complete_blocks * block_size_, input_dims_,
                    *remainder_matrix_);
  }

  auto final_span = MakeMutableSpan(*projected->mutable_values());
  auto do_block_round2 = [&](size_t block_start, size_t block_end,
                             const DenseDataset<float>& rotation_matrix) {
    const size_t current_block_size = block_end - block_start;
    DatapointPtr<FloatT> block = MakeDatapointPtr(
        MakeConstSpan(tmp_output).subspan(block_start, current_block_size));
    for (size_t offset_idx : Seq(current_block_size)) {
      const size_t global_dim_idx = block_start + offset_idx;
      final_span[global_dim_idx] =
          DotProduct(block, rotation_matrix[offset_idx]);
    }
  };

  for (size_t block_idx : Seq(num_complete_blocks)) {
    do_block_round2(block_idx * block_size_, (block_idx + 1) * block_size_,
                    *block_rotation_matrix_);
  }
  if (remainder_matrix_) {
    do_block_round2(num_complete_blocks * block_size_, input_dims_,
                    *remainder_matrix_);
  }

  return OkStatus();
}

template <typename T>
void BlockwiseRandomOrthogonalProjection<T>::MakePermutation() {
  permutation_ = vector<uint32_t>(input_dims_);
  std::iota(permutation_.begin(), permutation_.end(), uint32_t{0});
  std::shuffle(permutation_.begin(), permutation_.end(), *random_);
}

DEFINE_PROJECT_INPUT_OVERRIDES(BlockwiseRandomOrthogonalProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, BlockwiseRandomOrthogonalProjection);

}  // namespace research_scann
