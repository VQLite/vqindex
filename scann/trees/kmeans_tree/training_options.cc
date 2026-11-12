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



#include "scann/trees/kmeans_tree/training_options.h"

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/gmm_utils.h"

ABSL_FLAG(std::string, experimental_default_center_initialization_type,
          research_scann::ToString(
              research_scann::GmmUtils::Options().center_initialization_type),
          "The center initialization type to use for default KMeans tree "
          "training options instances. This is an experimental flag intended "
          "to be used for experiments only since this is not the expected way "
          "to set the center initialization type. This flag will be removed at "
          "any point in the future so production code should not rely on it.");

namespace research_scann {

std::string ToString(
    GmmUtils::Options::CenterInitializationType center_initialization_type) {
  switch (center_initialization_type) {
    case GmmUtils::Options::KMEANS_PLUS_PLUS:
      return "KMEANS_PLUS_PLUS";
    case GmmUtils::Options::MEAN_DISTANCE_INITIALIZATION:
      return "MEAN_DISTANCE_INITIALIZATION";
    case GmmUtils::Options::RANDOM_INITIALIZATION:
      return "RANDOM_INITIALIZATION";
    default:
      return "UNKNOWN";
  }
}

static const absl::NoDestructor<absl::flat_hash_map<
    absl::string_view, GmmUtils::Options::CenterInitializationType>>
    kCenterInitializationTypeMap(
        absl::flat_hash_map<absl::string_view,
                            GmmUtils::Options::CenterInitializationType>({
            {"KMEANS_PLUS_PLUS", GmmUtils::Options::KMEANS_PLUS_PLUS},
            {"MEAN_DISTANCE_INITIALIZATION",
             GmmUtils::Options::MEAN_DISTANCE_INITIALIZATION},
            {"RANDOM_INITIALIZATION", GmmUtils::Options::RANDOM_INITIALIZATION},
        }));

KMeansTreeTrainingOptions::KMeansTreeTrainingOptions() {
  const std::string target_type =
      absl::GetFlag(FLAGS_experimental_default_center_initialization_type);
  auto it = kCenterInitializationTypeMap->find(target_type);
  if (it != kCenterInitializationTypeMap->end()) {
    center_initialization_type = it->second;
  } else {
    LOG(WARNING) << "Unmapped center initialization type: " << target_type
                 << ". Using KMEANS_PLUS_PLUS instead.";
  }
}

KMeansTreeTrainingOptions::KMeansTreeTrainingOptions(
    const PartitioningConfig& config)
    : partitioning_type(config.partitioning_type()),
      max_num_levels(config.max_num_levels()),
      max_leaf_size(config.max_leaf_size()),
      learned_spilling_type(config.database_spilling().spilling_type()),
      per_node_spilling_factor(config.database_spilling().replication_factor()),
      max_spill_centers(config.database_spilling().max_spill_centers()),
      max_iterations(config.max_clustering_iterations()),
      convergence_epsilon(config.clustering_convergence_tolerance()),
      min_cluster_size(config.min_cluster_size()),
      balancing_num_nearest_centroids(config.balancing_num_nearest_centroids()),
      seed(config.clustering_seed()) {
  switch (config.balancing_type()) {
    case PartitioningConfig::DEFAULT_UNBALANCED:
      balancing_type = GmmUtils::Options::UNBALANCED;
      break;
    case PartitioningConfig::GREEDY_BALANCED:
      balancing_type = GmmUtils::Options::GREEDY_BALANCED;
      break;
    case PartitioningConfig::UNBALANCED_FLOAT32:
      balancing_type = GmmUtils::Options::UNBALANCED_FLOAT32;
      break;
    case PartitioningConfig::SINGLE_PASS_UNBALANCED_FLOAT32:
      balancing_type = GmmUtils::Options::SINGLE_PASS_UNBALANCED_FLOAT32;
      break;
    case PartitioningConfig::SINGLE_PASS_GREEDY_BALANCED:
      balancing_type = GmmUtils::Options::SINGLE_PASS_GREEDY_BALANCED;
      break;
  }
  switch (config.trainer_type()) {
    case PartitioningConfig::DEFAULT_SAMPLING_TRAINER:
    case PartitioningConfig::FLUME_KMEANS_TRAINER:
      reassignment_type = GmmUtils::Options::RANDOM_REASSIGNMENT;
      break;
    case PartitioningConfig::PCA_KMEANS_TRAINER:
    case PartitioningConfig::SAMPLING_PCA_KMEANS_TRAINER:
      reassignment_type = GmmUtils::Options::PCA_SPLITTING;
      break;
  }
  switch (config.single_machine_center_initialization()) {
    case PartitioningConfig::DEFAULT_KMEANS_PLUS_PLUS:
      center_initialization_type = GmmUtils::Options::KMEANS_PLUS_PLUS;
      break;
    case PartitioningConfig::RANDOM_INITIALIZATION:
      center_initialization_type = GmmUtils::Options::RANDOM_INITIALIZATION;
      break;
  }
}

}  // namespace research_scann
