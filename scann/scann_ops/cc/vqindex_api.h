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

// How vqindex_dump_format writes the AH codes of a trained index.
//  DUMP_FORMAT_PACKED  hashed_dataset_packed.bin: the codes as they are laid
//                      out in memory.  Loads and dumps without the extra
//                      copy the legacy file needs.  Not readable by
//                      libraries built before this format existed.
//  DUMP_FORMAT_LEGACY  hashed_dataset.npy, readable by every version.
//  DUMP_FORMAT_BOTH    both files.
// A dump removes the file of the form it does not write, so the index
// directory never holds codes from two different states.  Loading uses the
// packed file when the directory has one and the legacy file otherwise.
typedef enum {
    DUMP_FORMAT_PACKED = 0,
    DUMP_FORMAT_LEGACY = 1,
    DUMP_FORMAT_BOTH = 2
} dump_format_t;

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
};
typedef struct index_stats_s index_stats_t;

void *vqindex_init(const char *index_dir, index_config_t config_i);

void vqindex_release(void *vql_handler);

// Dumps in the default format: DUMP_FORMAT_PACKED, or the value (0, 1 or 2)
// of the environment variable VQINDEX_DUMP_FORMAT if it is set when the index
// is opened.  The dump at the end of vqindex_train_process uses it as well.
ret_code_t vqindex_dump(void *vql_handler);

// Dumps in the given format; RET_CODE_ERR if it is not a dump_format_t value.
ret_code_t vqindex_dump_format(void *vql_handler, dump_format_t format);

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

// len: number of queries float, <len % dim_ == 0>
// res: must hold (len / dim_) * params.topk_ items; params.topk_ must be > 0
// (0 selects the internal default of 30 and res must then hold 30 per query).
// Unused slots keep idx_ = 0, vid_ = 0, score_ = -1.
ret_code_t vqindex_search(
        void *vql_handler, const float *queries, int len, result_search_t *res, params_search_t params);

index_stats_t vqindex_stats(void *vql_handler);

#ifdef __cplusplus
}
#endif

#endif
