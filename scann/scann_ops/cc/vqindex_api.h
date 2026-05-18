/**
 * Copyright 2022 The VQLite Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef VQLITE_API_H_
#define VQLITE_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRAIN_TYPE_DEFAULT,
    TRAIN_TYPE_NEW,
    TRAIN_TYPE_ADD,
} train_type_t;

typedef enum {
    STORAGE_FILE,
    STORAGE_MEMORY
} storage_type_t;

typedef enum {
    INDEX_TYPE_SCANN,
    INDEX_TYPE_FAISS,
} index_type_t;

typedef enum {
    INDEX_STATE_NONE,
    INDEX_STATE_NOINIT,
    INDEX_STATE_NOINDEX,
    INDEX_STATE_READY,
    INDEX_STATE_ADD,
    INDEX_STATE_TRAIN,
    INDEX_STATE_DUMP
} index_state_t;

typedef enum {
    RET_CODE_OK = 0,
    RET_CODE_ERR = -1,
    RET_CODE_NOREADY = -2,
    RET_CODE_MEMORYERR = -3,
    RET_CODE_NOPERMISSION = -4,
    RET_CODE_DATAERR = -5,
    RET_CODE_INDEXERR = -6,
    RET_CODE_ADD2INDEXERR = -7,
    RET_CODE_NOINIT = -8
} ret_code_t;

typedef enum {
    ARTIFACT_FORMAT_UNKNOWN = 0,
    ARTIFACT_FORMAT_BRUTE = 1,
    ARTIFACT_FORMAT_LEAF_LUT16_PACKED = 2,
    ARTIFACT_FORMAT_LEGACY_HASHED = 3
} artifact_format_t;

struct index_config_s {
    index_type_t index_type_; // index type
    uint32_t dim_; // dimensions of vector point
    uint64_t brute_threshold_;

    float partitioning_train_sample_rate_; // default 0.2
    float hash_train_sample_rate_; // default 0.1
    storage_type_t storage_type_;
};
typedef struct index_config_s index_config_t;

struct params_search_s {
    uint32_t topk_; // final_nn
    uint32_t reorder_topk_; // pre_reorder_nn
    uint32_t nprobe_; // leaves_to_search
};
typedef struct params_search_s params_search_t;

struct result_search_s {
    uint64_t idx_;
    int64_t vid_;
    float score_;
};
typedef struct result_search_s result_search_t;

struct index_stats_s {
    int64_t datasets_size_;
    int64_t vid_size_;
    int64_t index_size_;
    int64_t brute_threshold_;
    int32_t index_nlist_;
    int32_t dim_;
    int8_t is_brute_;
    index_state_t current_status_;
    int64_t pending_size_;
    int64_t deleted_size_;
    int64_t last_load_ms_;
    int64_t last_dump_ms_;
    int64_t last_train_ms_;
    int64_t last_rebalance_ms_;
    artifact_format_t artifact_format_;
    int8_t use_autopilot_;
    int8_t enable_soar_;
};
typedef struct index_stats_s index_stats_t;

struct index_tuning_config_s {
    uint32_t use_autopilot_;
    uint32_t enable_soar_;
    uint32_t topk_;
    uint32_t reorder_topk_;
    uint32_t nprobe_;
    uint32_t autopilot_reordering_dtype_; // 0 keep default, 2 bf16, 3 int8
    float soar_lambda_;
    float soar_overretrieve_factor_;
    uint64_t autopilot_l1_size_;
    uint64_t autopilot_l3_size_;
};
typedef struct index_tuning_config_s index_tuning_config_t;

struct index_health_stats_s {
    double partition_weighted_avg_relative_imbalance_;
    double partition_avg_relative_positive_imbalance_;
    double avg_quantization_error_;
    uint64_t sum_partition_sizes_;
};
typedef struct index_health_stats_s index_health_stats_t;

void *vqindex_init(const char *index_dir, index_config_t config_i);

void vqindex_release(void *vql_handler);

ret_code_t vqindex_dump(void *vql_handler);

ret_code_t vqindex_flush(void *vql_handler);

// if nlist=0, use default nlist, it's only available to New[train_type].
// nlist: number of partitioning leaves, If a dataset has n points,
// the number of partitions should generally be the same order
// of magnitude as sqrt(n) for a good balance of partitioning quality and
// speed. num_leaves_to_search should be tuned based on recall target.
ret_code_t vqindex_train(
        void *vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads);

// use process to train
ret_code_t vqindex_train_process(
        void *vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads);


// len: number of datasets float, <npoint = dim_ / len>, <len % dim_ == 0>.
// only add to datasets, not index.
ret_code_t vqindex_add(
        void *vql_handler, const float *datasets, uint64_t len, const int64_t *vids);

ret_code_t vqindex_insert(
        void *vql_handler, const float *datasets, uint64_t len, const int64_t *vids);

ret_code_t vqindex_upsert(
        void *vql_handler, const float *datasets, uint64_t len, const int64_t *vids);

ret_code_t vqindex_delete(void *vql_handler, const int64_t *vids, uint64_t n);

// len: number of queries float, <len % dim_ == 0>
ret_code_t vqindex_search(
        void *vql_handler, const float *queries, uint64_t len, result_search_t *res,
        uint64_t res_capacity, uint64_t *res_count, params_search_t params);

index_stats_t vqindex_stats(void *vql_handler);

ret_code_t vqindex_last_error(void *vql_handler, char *error_msg, uint64_t error_msg_len);

ret_code_t vqindex_clear_error(void *vql_handler);

ret_code_t vqindex_version(char *version, uint64_t version_len);

ret_code_t vqindex_capabilities(char *capabilities, uint64_t capabilities_len);

ret_code_t vqindex_set_tuning(void *vql_handler, index_tuning_config_t tuning);

ret_code_t vqindex_suggest_config(
        void *vql_handler, uint64_t dataset_size, uint32_t nlist,
        char *config_pbtxt, uint64_t config_pbtxt_len);

ret_code_t vqindex_current_config(
        void *vql_handler, char *config_pbtxt, uint64_t config_pbtxt_len);

ret_code_t vqindex_rebalance(void *vql_handler, const char *config_pbtxt, int32_t nthreads);

ret_code_t vqindex_initialize_health_stats(void *vql_handler);

ret_code_t vqindex_health_stats(void *vql_handler, index_health_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
