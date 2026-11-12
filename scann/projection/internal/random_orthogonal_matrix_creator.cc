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

#include "scann/projection/internal/random_orthogonal_matrix_creator.h"

#include "Eigen/Core"
#include "Eigen/QR"
#include "absl/random/random.h"

namespace research_scann {

unique_ptr<DenseDataset<float>> CreateRandomOrthogonalMatrix(
    int32_t input_dims, int32_t projected_dims, MTRandom& random) {
  Eigen::MatrixXf input_matrix(input_dims, projected_dims);

  for (size_t i : Seq(projected_dims)) {
    for (size_t j : Seq(input_dims)) {
      input_matrix(j, i) = absl::Gaussian<double>(random);
    }
  }

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(input_matrix);
  Eigen::MatrixXf Q = qr.householderQ();

  auto random_rotation_matrix = std::make_unique<DenseDataset<float>>();
  for (size_t i : Seq(projected_dims)) {
    vector<float> current(input_dims);
    for (size_t j : Seq(input_dims)) {
      current[j] = Q(j, i);
    }
    random_rotation_matrix->AppendOrDie(MakeDatapointPtr(current), "");
  }
  return random_rotation_matrix;
}

}  // namespace research_scann
