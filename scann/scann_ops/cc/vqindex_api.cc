/**
 * Copyright 2022 The VQLite Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "scann/scann_ops/cc/vqindex_api.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cmath>
#include <memory>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include <chrono>
#include <thread>
#include<sys/wait.h>

#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/packed_leaves_file.h"
#include "scann/utils/threads.h"
#include "scann/utils/types.h"

using namespace research_scann;
using namespace std;

namespace vqindex {

class VQLiteIndex {
public:
    VQLiteIndex(const char* index_root_dir, index_config_t config_i)
        : dim_(config_i.dim_)
        , datasets_filename_("datasets.vql")
        , vids_filename_("vids.vql")
        , index_subdir_name_("index")
        , current_state_(INDEX_STATE_NOINIT)
        , current_search_n_(0)
        , datasets_npoints_(0)
        , is_child_process_(false)
        , stat_index_npoints_(0)
        , stat_index_nlist_(0)
        , stat_is_brute_(false)
        , persisted_datasets_floats_(0)
        , persisted_vids_(0)
    {
        default_dump_format_ = DefaultDumpFormat();
        index_root_dir_ = ".";
        if (index_root_dir != NULL) {
            index_root_dir_ = index_root_dir;
        }
        storage_type_ = STORAGE_FILE;
        if (config_i.storage_type_ >= STORAGE_FILE && config_i.storage_type_ <= STORAGE_MEMORY) {
            storage_type_ = config_i.storage_type_;
        }

        brute_threshold_ = 4096;
        if (config_i.brute_threshold_ > 0 && config_i.brute_threshold_ < 40960) {
            brute_threshold_ = config_i.brute_threshold_;
        }

        std::error_code ec;
        if (!IsExists(index_root_dir_)
            && !std::filesystem::create_directories(index_root_dir_, ec)) {
            LOG(INFO) << "Create " + index_root_dir_ + " Directories Fail.";
        }
    }

    virtual ~VQLiteIndex() { }

    bool IsExists(std::string& file_path) { return std::filesystem::exists(file_path); }

    int64_t GetFileSize(std::string& file_path)
    {
        if (!IsExists(file_path)) {
            return 0;
        }
        return std::filesystem::file_size(file_path);
    }

    // Lock-free apart from a short shared lock on the vid table, so that it
    // can be polled while a train/add/dump is running.  The index figures come
    // from the snapshot taken by RefreshIndexStats(); the index object itself
    // must not be touched here because Train() may be replacing it.
    void GetStats(index_stats_t& stats)
    {
        stats.datasets_size_ = datasets_npoints_.load();
        stats.index_size_ = stat_index_npoints_.load();
        {
            shared_lock<shared_mutex> rslk(smutex_vid_rwlock_);
            stats.vid_size_ = vids_.size();
        }
        stats.index_nlist_ = stat_index_nlist_.load();
        stats.dim_ = dim_;
        if (stat_is_brute_.load()) {
            stats.is_brute_ = 1;
        }
        stats.brute_threshold_ = brute_threshold_;
        stats.current_status_ = current_state_.load();
    }

    bool LoadVids()
    {
        std::string vids_file_path = index_root_dir_ + "/" + vids_filename_;
        if (!IsExists(vids_file_path)) {
            LOG(INFO) << "LoadVids: " << vids_file_path;
            return true;
        }
        std::ifstream in(vids_file_path, ios::in | std::ifstream::ate | std::ifstream::binary);
        if (!in.is_open()) {
            return false;
        }

        int64_t file_vid_size = in.tellg();
        in.seekg(0, ios::beg);
        vids_.resize(file_vid_size / sizeof(int64_t));
        in.read((char*)(vids_.data()), file_vid_size);
        in.close();

        LOG(INFO) << "vids size=" << vids_.size();

        return true;
    }

    bool LoadTrainDatasets(std::vector<float>& datasets, size_t offset_npoints)
    {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        if (!IsExists(datasets_file_path)) {
            return false;
        }
        std::ifstream in(datasets_file_path, ios::in | std::ifstream::ate | std::ifstream::binary);
        if (!in.is_open()) {
            return false;
        }
        int64_t file_dataset_size = in.tellg();
        int64_t npoints_datasets = file_dataset_size / sizeof(float) / dim_;
        int64_t npoints_datasets_add = npoints_datasets - offset_npoints;
        int64_t datasets_add_size = npoints_datasets_add * sizeof(float) * dim_;

        int64_t start_position = offset_npoints * dim_ * sizeof(float);
        in.seekg(start_position, ios::beg);
        datasets.resize(datasets_add_size / sizeof(float));
        in.read((char*)(datasets.data()), datasets_add_size);
        in.close();

        return true;
    }

    bool Init();
    virtual bool InitImpl(std::string& index_dir)
    {
        LOG(INFO) << "InitImpl Unavailable";
        return 0;
    }

    // only add vectors to original datasets
    ret_code_t AddDatasets(const float* datasets, uint64_t len, const int64_t* vids);

    void Reset()
    {
        std::vector<float> t1;
        datasets_.swap(t1);
        std::vector<int64_t> t2;
        vids_.swap(t2);

        datasets_npoints_ = 0;
        persisted_datasets_floats_ = 0;
        persisted_vids_ = 0;

        ResetImpl();
        RefreshIndexStats();
    }
    virtual void ResetImpl() { LOG(INFO) << "ResetImpl Unavailable"; }

    // add datasets to index, if the index already exists.
    int Add(std::vector<float>& datasets, int32_t nthreads);
    virtual int AddImpl(std::vector<float>& datasets, uint64_t npoints, int32_t nthreads)
    {
        LOG(INFO) << "AddImpl Unavailable";
        return 0;
    }

    ret_code_t Train(train_type_t train_type, uint32_t nlist, int32_t nthreads);
    ret_code_t TrainProcess(train_type_t train_type, uint32_t nlist, int32_t nthreads);
    ret_code_t TrainInChildProcess(train_type_t train_type, uint32_t nlist, int32_t nthreads);
    ret_code_t RunTrain(train_type_t train_type, uint32_t nlist, int32_t nthreads);
    ret_code_t TrainDefault(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainNew(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainAdd(int32_t nthreads);
    virtual int TrainImpl(
        std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
    {
        LOG(INFO) << "TrainImpl Unavailable";
        return 0;
    }

    static bool IsDumpFormat(int format)
    {
        return format == DUMP_FORMAT_PACKED || format == DUMP_FORMAT_LEGACY
            || format == DUMP_FORMAT_BOTH;
    }
    static dump_format_t DefaultDumpFormat()
    {
        const char* env = getenv("VQINDEX_DUMP_FORMAT");
        if (env != NULL && env[0] >= '0' && env[0] <= '2' && env[1] == '\0') {
            return static_cast<dump_format_t>(env[0] - '0');
        }
        if (env != NULL && env[0] != '\0') {
            LOG(INFO) << "ignoring VQINDEX_DUMP_FORMAT=" << env << " (expected 0, 1 or 2)";
        }
        return DUMP_FORMAT_PACKED;
    }

    ret_code_t Dump() { return Dump(default_dump_format_); }
    ret_code_t Dump(dump_format_t format);
    ret_code_t DoDump(dump_format_t format);
    virtual int DumpImpl(std::string& index_dir, dump_format_t format)
    {
        LOG(INFO) << "DumpImpl Unavailable";
        return 0;
    }

    ret_code_t Search(const float* queries, int32_t len, std::vector<result_search_t>& res,
        params_search_t params);
    // counts[i] is the number of valid results of query i; they occupy the
    // first counts[i] of its topk slots in res.
    virtual int SearchImpl(const float* queries, int32_t npoints, std::vector<result_search_t>& res,
        std::vector<uint32_t>& counts, params_search_t params)
    {
        LOG(INFO) << "SearchImpl Unavailable";
        return 0;
    }

    uint32_t GetDim() { return dim_; }
    virtual size_t GetPartitioningSize() = 0;
    virtual size_t GetIndexPointsNum() = 0;

    virtual bool IsBrute() { return false; }

protected:
    ret_code_t AddDatasetsMemory(const float* datasets, uint64_t len, const int64_t* vids);
    ret_code_t AddDatasetsFile(const float* datasets, uint64_t len, const int64_t* vids);

    // Snapshots the index figures reported by GetStats().  Call with exclusive
    // access to the index (Init, or mutex_global_lock_ held).
    void RefreshIndexStats()
    {
        stat_index_npoints_ = GetIndexPointsNum();
        stat_index_nlist_ = GetPartitioningSize();
        stat_is_brute_ = IsBrute();
    }

    // Leaves the ADD/TRAIN/DUMP state.  Same access requirement as above.
    void RestoreState()
    {
        RefreshIndexStats();
        if (stat_index_npoints_.load() > 0) {
            current_state_ = INDEX_STATE_READY;
        } else {
            current_state_ = INDEX_STATE_NOINDEX;
        }
    }

    // Brings `path` up to `total_bytes` of `data`.  When the file still holds
    // exactly the first `persisted_bytes` written or loaded earlier, only the
    // tail is appended; otherwise the whole file is rewritten.
    bool WriteTailOrAll(
        std::string& path, const char* data, size_t total_bytes, size_t persisted_bytes)
    {
        bool append = persisted_bytes > 0 && persisted_bytes <= total_bytes
            && GetFileSize(path) == (int64_t)persisted_bytes;
        std::ofstream outs(
            path, append ? (ios::app | ios::binary) : (ios::out | ios::trunc | ios::binary));
        if (!outs) {
            return false;
        }
        size_t offset = append ? persisted_bytes : 0;
        outs.write(data + offset, total_bytes - offset);
        outs.close();
        return !outs.fail();
    }

    std::string index_root_dir_;
    std::string datasets_filename_;
    std::string vids_filename_;
    std::string index_subdir_name_;

    std::vector<float> datasets_;
    std::vector<int64_t> vids_;

    std::atomic<uint64_t> datasets_npoints_;

    uint32_t dim_; // dimensions of vector point
    storage_type_t storage_type_;
    uint64_t brute_threshold_;

    // Locking:
    //  * mutex_global_lock_ serializes AddDatasets / Train / Dump.  The
    //    ADD/TRAIN/DUMP states are only entered and left while holding it.
    //    Add and Train do not queue behind a running operation, they return
    //    RET_CODE_NOREADY; Dump waits.
    //  * Search takes no mutex for the index itself.  It registers in
    //    current_search_n_ *before* reading current_state_, and Train sets
    //    INDEX_STATE_TRAIN *before* waiting for current_search_n_ to reach 0,
    //    so a search either backs off or is waited for before the index is
    //    replaced or mutated.
    //  * smutex_vid_rwlock_ guards vids_ between a writer (AddDatasets, which
    //    also holds mutex_global_lock_) and lock-free readers (Search,
    //    GetStats).  Order: mutex_global_lock_ -> smutex_vid_rwlock_.
    std::mutex mutex_global_lock_;
    std::shared_mutex smutex_vid_rwlock_;
    std::atomic<int> current_search_n_;

    bool is_child_process_;

    // Format used by vqindex_dump and by the dump after a child-process train.
    dump_format_t default_dump_format_;

    std::atomic<index_state_t> current_state_;

    // Snapshot for GetStats(), see RefreshIndexStats().
    std::atomic<uint64_t> stat_index_npoints_;
    std::atomic<uint32_t> stat_index_nlist_;
    std::atomic<bool> stat_is_brute_;

    // STORAGE_MEMORY: how much of datasets_ / vids_ is known to be on disk
    // (loaded by Init or written by Dump), so Dump only appends what is new.
    uint64_t persisted_datasets_floats_;
    uint64_t persisted_vids_;
};

bool VQLiteIndex::Init()
{
    bool ret = false;

    std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
    int64_t datasets_filesize = GetFileSize(datasets_file_path);
    int64_t datasets_npoints = datasets_filesize / sizeof(float) / dim_;
    size_t index_npoints = 0;

    std::string index_dir = index_root_dir_ + '/' + index_subdir_name_;
    if (LoadVids() && InitImpl(index_dir)) {
        index_npoints = GetIndexPointsNum();
        if (vids_.size() <= datasets_npoints + index_npoints) {
            ret = true;
        }
        LOG(INFO) << "vids_size=" << vids_.size() << "; index_npoints=" << index_npoints
                  << "; datasets_npoints=" << datasets_npoints;
    }
    if (ret && storage_type_ == STORAGE_MEMORY) {
        LoadTrainDatasets(datasets_, 0);
        LOG(INFO) << "dataset_.size=" << datasets_.size();
    }
    if (!ret) {
        Reset();
    } else {
        datasets_npoints_ = datasets_npoints;
        if (storage_type_ == STORAGE_MEMORY) {
            persisted_datasets_floats_ = datasets_.size();
            persisted_vids_ = vids_.size();
        }
        RestoreState();
    }

    return ret;
}

ret_code_t VQLiteIndex::AddDatasetsMemory(const float* datasets, uint64_t len, const int64_t* vids)
{
    uint64_t add_npoints = len / dim_;
    uint64_t now_npoints = datasets_.size() / dim_;
    uint64_t new_npoints = now_npoints + add_npoints;

    datasets_.resize(new_npoints * dim_);
    memcpy(datasets_.data() + now_npoints * dim_, datasets, len * sizeof(float));

    return RET_CODE_OK;
}

ret_code_t VQLiteIndex::AddDatasetsFile(const float* datasets, uint64_t len, const int64_t* vids)
{
    uint64_t add_npoints = len / dim_;
    std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
    std::string vids_file_path = index_root_dir_ + "/" + vids_filename_;

    std::ofstream outs_datasets(datasets_file_path, ios::app | ios::binary);
    if (!outs_datasets) {
        return RET_CODE_NOPERMISSION;
    }
    outs_datasets.write((char*)datasets, len * sizeof(float));
    outs_datasets.close();
    if (outs_datasets.bad()) {
        LOG(INFO) << "Add Datasets File Fail.";
        return RET_CODE_NOPERMISSION;
    }

    std::ofstream outs_vids(vids_file_path, ios::app | ios::binary);
    if (!outs_vids) {
        return RET_CODE_NOPERMISSION;
    }
    outs_vids.write((char*)vids, add_npoints * sizeof(int64_t));
    outs_vids.close();
    if (outs_vids.bad()) {
        LOG(INFO) << "Add vids File Fail.";
        return RET_CODE_NOPERMISSION;
    }

    return RET_CODE_OK;
}

ret_code_t VQLiteIndex::AddDatasets(const float* datasets, uint64_t len, const int64_t* vids)
{
    if (len % dim_ != 0) {
        LOG(INFO) << "len % dim_ != 0; len=" << len;
        return RET_CODE_DATAERR;
    }

    // Everything below, including reading the current sizes, has to happen
    // under the lock: two concurrent adds must not start from the same offset.
    std::unique_lock<std::mutex> guard(mutex_global_lock_, std::try_to_lock);
    if (!guard.owns_lock() || current_state_ > INDEX_STATE_READY) {
        return RET_CODE_NOREADY;
    }

    current_state_ = INDEX_STATE_ADD;

    uint64_t add_npoints = len / dim_;
    uint64_t now_npoints = vids_.size();
    uint64_t new_npoints = now_npoints + add_npoints;

    if (vids_.capacity() < new_npoints) {
        // Reallocation moves the table under concurrent readers.
        unique_lock<shared_mutex> wulk(smutex_vid_rwlock_);
        size_t new_capacity = vids_.capacity() + add_npoints + 400000;
        vids_.reserve(new_capacity);
    }

    ret_code_t ret = RET_CODE_OK;
    if (storage_type_ == STORAGE_MEMORY) {
        ret = AddDatasetsMemory(datasets, len, vids);
    } else {
        ret = AddDatasetsFile(datasets, len, vids);
    }

    if (ret == RET_CODE_OK) {
        {
            unique_lock<shared_mutex> wulk(smutex_vid_rwlock_);
            vids_.resize(new_npoints, 0);
            memcpy(vids_.data() + now_npoints, vids, add_npoints * sizeof(int64_t));
        }
        datasets_npoints_ += add_npoints;
    }
    LOG(INFO) << "vids_.size=" << vids_.size() << "; dataset.size=" << datasets_.size()
              << "; vids_.capacity()=" << vids_.capacity();

    RestoreState();

    return ret;
}

int VQLiteIndex::Add(std::vector<float>& datasets, int32_t nthreads)
{
    if (datasets.size() % dim_ != 0) {
        return -1;
    }

    uint64_t add_npoints = datasets.size() / dim_;
    return AddImpl(datasets, add_npoints, nthreads);
}

ret_code_t VQLiteIndex::TrainDefault(uint32_t nlist, int32_t nthreads)
{
    int64_t datasets_npoints = 0;
    int64_t vids_npoints = vids_.size();
    int64_t index_npoints = GetIndexPointsNum();
    int32_t index_nlist = GetPartitioningSize();

    if (index_npoints <= 0 || IsBrute()) {
        LOG(INFO) << "index_npoints <= 0 || IsBrute()  TrainNew";
        return TrainNew(nlist, nthreads);
    }

    if (storage_type_ == STORAGE_FILE) {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        int64_t datasets_filesize = GetFileSize(datasets_file_path);
        datasets_npoints = datasets_filesize / sizeof(float) / dim_;
    } else {
        datasets_npoints = datasets_.size() / dim_;
    }

    if (datasets_npoints != vids_npoints || datasets_npoints < (int64_t)index_nlist * index_nlist * 2) {
        LOG(INFO) << "datasets_npoints != vids_npoints  TrainAdd: datasets_npoints="
                  << datasets_npoints << "; vids_npoints=" << vids_npoints;
        LOG(INFO) << "index_npoints=" << index_npoints << ";index_nlist=" << index_nlist;
        return TrainAdd(nthreads);
    }

    return TrainNew(nlist, nthreads);
}

ret_code_t VQLiteIndex::TrainNew(uint32_t nlist, int32_t nthreads)
{
    std::vector<float> datasets, *datasets_ptr = NULL;
    if (storage_type_ == STORAGE_FILE) {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        int64_t datasets_filesize = GetFileSize(datasets_file_path);
        int64_t datasets_npoints = datasets_filesize / sizeof(float) / dim_;

        int64_t vids_npoints = vids_.size();

        if (vids_npoints != datasets_npoints) {
            LOG(INFO) << "vids_npoints=" << vids_npoints
                      << "; datasets_npoints=" << datasets_npoints;
            return RET_CODE_DATAERR;
        }
        LoadTrainDatasets(datasets, 0);
        if (datasets.size() / dim_ != datasets_npoints) {
            LOG(INFO) << "Load Train Datasets Fail.";
            return RET_CODE_DATAERR;
        }
        datasets_ptr = &datasets;
    } else {
        if (datasets_.size() / dim_ != vids_.size()) {
            LOG(INFO) << "Memory Datasets Size Error.";
            return RET_CODE_DATAERR;
        }
        datasets_ptr = &datasets_;
    }

    if (TrainImpl(*datasets_ptr, datasets_ptr->size() / dim_, nlist, nthreads) != 0) {
        return RET_CODE_DATAERR;
    }
    LOG(INFO) << "datasets_ptr size=" << datasets_ptr->size();

    return RET_CODE_OK;
}

ret_code_t VQLiteIndex::TrainAdd(int32_t nthreads)
{
    size_t npoints_index = GetIndexPointsNum();
    int64_t vids_npoints = vids_.size();
    if (npoints_index == vids_npoints) {
        return RET_CODE_OK;
    }
    if (npoints_index <= 0) {
        return RET_CODE_NOREADY;
    }
    std::vector<float> datasets, *datasets_ptr = NULL;
    if (storage_type_ == STORAGE_FILE) {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        int64_t datasets_filesize = GetFileSize(datasets_file_path);
        int64_t datasets_npoints = datasets_filesize / sizeof(float) / dim_;
        size_t npoints_offset = 0, datasets_add_npoints = 0;

        if (vids_npoints > npoints_index + datasets_npoints) {
            LOG(INFO) << "vids_npoints=" << vids_npoints
                      << "; datasets_npoints=" << datasets_npoints;
            return RET_CODE_DATAERR;
        }

        datasets_add_npoints = vids_npoints - npoints_index;
        npoints_offset = datasets_npoints - datasets_add_npoints;

        LOG(INFO) << "datasets_add_npoints=" << datasets_add_npoints
                  << "; npoints_offset=" << npoints_offset;
        LoadTrainDatasets(datasets, npoints_offset);
        LOG(INFO) << "datasets.size() / dim_=" << datasets.size() / dim_;
        if (datasets.size() / dim_ != datasets_add_npoints) {
            LOG(INFO) << "Load Train Datasets Fail.";
            return RET_CODE_DATAERR;
        }
        datasets_ptr = &datasets;
    } else {
        int64_t datasets_npoints = datasets_.size() / dim_;
        size_t npoints_offset = 0, datasets_add_npoints = 0;
        if (vids_npoints > npoints_index + datasets_npoints) {
            LOG(INFO) << "vids_npoints=" << vids_npoints
                      << "; datasets_npoints=" << datasets_npoints;
            return RET_CODE_DATAERR;
        }

        datasets_add_npoints = vids_npoints - npoints_index;
        npoints_offset = datasets_npoints - datasets_add_npoints;
        LOG(INFO) << "datasets_add_npoints=" << datasets_add_npoints
                  << "; npoints_offset=" << npoints_offset;
        datasets.resize(datasets_add_npoints * dim_);
        memcpy(datasets.data(), datasets_.data() + npoints_offset * dim_,
            datasets_add_npoints * sizeof(float) * dim_);
        datasets_ptr = &datasets;
    }

    if (Add(*datasets_ptr, nthreads) != 0) {
        return RET_CODE_ADD2INDEXERR;
    }
    return RET_CODE_OK;
}

ret_code_t VQLiteIndex::RunTrain(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    switch (train_type) {
    case TRAIN_TYPE_NEW:
        return TrainNew(nlist, nthreads);
    case TRAIN_TYPE_ADD:
        return TrainAdd(nthreads);
    default:
        return TrainDefault(nlist, nthreads);
    }
}

ret_code_t VQLiteIndex::Train(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    std::unique_lock<std::mutex> guard(mutex_global_lock_, std::try_to_lock);
    if (!guard.owns_lock() || current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "busy, current_state_=" << current_state_.load();
        return RET_CODE_NOREADY;
    }

    // From here on new searches back off (see Search); wait for the ones that
    // registered before the state change.
    current_state_ = INDEX_STATE_TRAIN;

    for (int waited_ms = 0; current_search_n_ > 0; waited_ms += 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (waited_ms % 1000 == 990) {
            LOG(INFO) << "wait search = " << current_search_n_;
        }
    }

    ret_code_t ret = RunTrain(train_type, nlist, nthreads);

    RestoreState();

    return ret;
}

// Runs in the forked child, which has no other threads: no locking.
ret_code_t VQLiteIndex::TrainProcess(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    is_child_process_ = true;

    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_.load();
        return RET_CODE_NOREADY;
    }

    current_state_ = INDEX_STATE_TRAIN;

    ret_code_t ret = RunTrain(train_type, nlist, nthreads);

    RestoreState();

    return ret;
}

// Trains and dumps in a forked child.  This object is not updated: the caller
// picks the new index up by releasing the handler and calling init again.
ret_code_t VQLiteIndex::TrainInChildProcess(
    train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    pid_t pid = 0;
    {
        // Fork while holding the lock so that the child never inherits a
        // half-applied add/train/dump.  The lock is not kept while waiting.
        std::unique_lock<std::mutex> guard(mutex_global_lock_, std::try_to_lock);
        if (!guard.owns_lock() || current_state_ > INDEX_STATE_READY) {
            LOG(INFO) << "busy, current_state_=" << current_state_.load();
            return RET_CODE_NOREADY;
        }

        pid = fork();
        if (pid == 0) {
            // The child owns a copy of the locked mutex and never uses it
            // (is_child_process_); it must not unlock it either.
            guard.release();
            ret_code_t cret_code = TrainProcess(train_type, nlist, nthreads);
            if (cret_code == RET_CODE_OK) {
                LOG(INFO) << "train suss, in dump.";
                cret_code = Dump();
            }
            LOG(INFO) << "out child process, cret_code=" << cret_code;
            // _exit: do not run the parent's atexit handlers / static
            // destructors, nor flush its stdio buffers a second time.
            // ret_code_t is 0 or negative, exit statuses are 0..255.
            _exit(-(int)cret_code);
        }
    }
    if (pid == -1) {
        LOG(INFO) << "fork error: " << strerror(errno);
        return RET_CODE_ERR;
    }

    LOG(INFO) << "in parent process, waitting child " << pid;
    int child_status = 0;
    while (waitpid(pid, &child_status, 0) == -1) {
        if (errno != EINTR) {
            LOG(INFO) << "waitpid error: " << strerror(errno);
            return RET_CODE_ERR;
        }
    }

    ret_code_t ret_code = RET_CODE_ERR;
    if (WIFEXITED(child_status)) {
        int code = WEXITSTATUS(child_status);
        if (code >= 0 && code <= -(int)RET_CODE_NOINIT) {
            ret_code = (ret_code_t)(-code);
        }
    } else if (WIFSIGNALED(child_status)) {
        LOG(INFO) << "child killed by signal " << WTERMSIG(child_status);
    }
    LOG(INFO) << "out parent process, ret_code=" << ret_code << "; child_status=" << child_status;

    return ret_code;
}

ret_code_t VQLiteIndex::DoDump(dump_format_t format)
{
    std::error_code ec;
    std::string index_dir = index_root_dir_ + "/" + index_subdir_name_;
    ret_code_t ret = RET_CODE_OK;

    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_.load();
        return RET_CODE_NOREADY;
    }

    current_state_ = INDEX_STATE_DUMP;

    if (!IsExists(index_dir) && !std::filesystem::create_directories(index_dir, ec)) {
        LOG(INFO) << "Create Directories Fail.";
        ret = RET_CODE_NOPERMISSION;
        goto end;
    }
    if (DumpImpl(index_dir, format) != 0) {
        LOG(INFO) << "Dump Index Fail.";
        ret = RET_CODE_INDEXERR;
        goto end;
    }
    if (storage_type_ == STORAGE_MEMORY) {
        // Both files only ever grow, so only what was added since the last
        // load/dump is written.
        std::string vids_file_path = index_root_dir_ + "/" + vids_filename_;
        if (!WriteTailOrAll(vids_file_path, (const char*)vids_.data(),
                vids_.size() * sizeof(int64_t), persisted_vids_ * sizeof(int64_t))) {
            LOG(INFO) << "Dump vids Fail.";
            persisted_vids_ = 0;
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
        persisted_vids_ = vids_.size();

        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        if (!WriteTailOrAll(datasets_file_path, (const char*)datasets_.data(),
                datasets_.size() * sizeof(float), persisted_datasets_floats_ * sizeof(float))) {
            LOG(INFO) << "Dump Datasets Fail.";
            persisted_datasets_floats_ = 0;
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
        persisted_datasets_floats_ = datasets_.size();
    }

end:
    RestoreState();
    return ret;
}

ret_code_t VQLiteIndex::Dump(dump_format_t format)
{
    if (!IsDumpFormat(format)) {
        LOG(INFO) << "unknown dump format " << static_cast<int>(format);
        return RET_CODE_ERR;
    }
    if (this->is_child_process_) {
        return this->DoDump(format);
    }
    std::lock_guard<std::mutex> guard(mutex_global_lock_);
    return this->DoDump(format);
}

ret_code_t VQLiteIndex::Search(
    const float* queries, int32_t len, std::vector<result_search_t>& res, params_search_t params)
{
    int32_t npoints = len / dim_;
    if (len % dim_ != 0) {
        LOG(INFO) << "Query Len % Dim != 0";
        return RET_CODE_DATAERR;
    }

    // Register first, then look at the state: Train() does the opposite (sets
    // the state, then waits for the counter), so it can never replace the index
    // under a search that got past this check.
    current_search_n_++;
    index_state_t state = current_state_.load();
    if (state == INDEX_STATE_TRAIN || state < INDEX_STATE_READY) {
        current_search_n_--;
        LOG(INFO) << "current_state_ = INDEX_STATE_TRAIN|INDEX_STATE_NOINDEX";
        return RET_CODE_NOREADY;
    }

    std::vector<uint32_t> counts;
    int ret_s = SearchImpl(queries, npoints, res, counts, params);
    current_search_n_--;

    if (ret_s != 0) {
        LOG(INFO) << "SearchImpl Fail: " << ret_s;
        if (ret_s == -1) {
            return RET_CODE_NOREADY;
        }
        if (ret_s == -2) {
            return RET_CODE_INDEXERR;
        }
        return RET_CODE_ERR;
    }

    // Valid results are identified by their count, not by the sign of the
    // score: a real hit can have a negative inner product.
    size_t topk = counts.empty() ? 0 : res.size() / counts.size();
    shared_lock<shared_mutex> rslk(smutex_vid_rwlock_);
    for (size_t i : Seq(counts.size())) {
        for (size_t j = 0; j < counts[i] && j < topk; j++) {
            result_search_t& item = res[i * topk + j];
            if (item.idx_ < vids_.size()) {
                item.vid_ = vids_[item.idx_];
            }
        }
    }
    return RET_CODE_OK;
}

class VQLiteIndexScann : public VQLiteIndex {
public:
    VQLiteIndexScann(const char* index_dir, index_config_t config_i)
        : VQLiteIndex(index_dir, config_i)
        , scann_handler_(NULL)
        , partitioning_train_sample_rate_(config_i.partitioning_train_sample_rate_)
        , hash_train_sample_rate_(config_i.hash_train_sample_rate_)
        , topk_(30)
        , reorder_topk_(256)
        , nprobe_(128)
        , is_brute_(false)
    {
        if (partitioning_train_sample_rate_ < 0 || partitioning_train_sample_rate_ > 1) {
            partitioning_train_sample_rate_ = 0.2;
        }
        if (hash_train_sample_rate_ < 0 || hash_train_sample_rate_ > 1) {
            hash_train_sample_rate_ = 0.1;
        }
    }

    ~VQLiteIndexScann()
    {
        if (scann_handler_ != NULL) {
            delete scann_handler_;
        }
    }

    inline std::string ReadFileString(const string& filename)
    {
        std::ifstream in(filename);
        if (!in) {
            return std::string();
        }
        std::istreambuf_iterator<char> begin(in), end;
        string content(begin, end);
        in.close();
        return content;
    }

    std::string GetAssetsInfo(std::string& index_dir, bool packed_hashes)
    {
        std::string assets_info = "assets { \n\
                asset_type: AH_CENTERS \n\
                asset_path: \"$index_dir/ah_codebook.pb\" \n\
            } \n\
            assets { \n\
                asset_type: PARTITIONER \n\
                asset_path: \"$index_dir/serialized_partitioner.pb\" \n\
            }\n\
            assets { \n\
                asset_type: TOKENIZATION_NPY \n\
                asset_path: \"$index_dir/datapoint_to_token.npy\" \n\
            } \n\
            assets { \n\
                asset_type: $hashed_type \n\
                asset_path: \"$index_dir/$hashed_file\" \n\
            } \n\
            assets { \n\
                asset_type: INT8_DATASET_NPY \n\
                asset_path: \"$index_dir/int8_dataset.npy\" \n\
            } \n\
            assets { \n\
                asset_type: INT8_MULTIPLIERS_NPY \n\
                asset_path: \"$index_dir/int8_multipliers.npy\" \n\
            } \n\
            assets { \n\
                asset_type: INT8_NORMS_NPY \n\
                asset_path: \"$index_dir/dp_norms.npy\" \n\
            }";

        std::string assets_info_brute = "assets { \n\
                asset_type: DATASET_NPY \n\
                asset_path: \"$index_dir/dataset.npy\" \n\
            }";

        if (is_brute_) {
            return std::regex_replace(assets_info_brute, std::regex("\\$index_dir"), index_dir);
        }
        assets_info = std::regex_replace(assets_info, std::regex("\\$hashed_type"),
            packed_hashes ? "AH_DATASET_PACKED_LEAVES" : "AH_DATASET_NPY");
        assets_info = std::regex_replace(assets_info, std::regex("\\$hashed_file"),
            packed_hashes ? kPackedLeavesFilename : "hashed_dataset.npy");
        return std::regex_replace(assets_info, std::regex("\\$index_dir"), index_dir);
    }

    bool CheckIndexFiles(std::string& index_dir)
    {
        std::string index_files[] = { "scann_config.pb", "ah_codebook.pb",
            "serialized_partitioner.pb", "datapoint_to_token.npy",
            "int8_dataset.npy", "int8_multipliers.npy", "dp_norms.npy" };

        // The AH codes come in either form (see dump_format_t).
        std::string packed_file = index_dir + "/" + kPackedLeavesFilename;
        std::string legacy_file = index_dir + "/hashed_dataset.npy";
        has_packed_hashes_ = IsExists(packed_file);
        has_legacy_hashes_ = IsExists(legacy_file);

        bool ret = has_packed_hashes_ || has_legacy_hashes_;
        for (size_t i : Seq(sizeof(index_files) / sizeof(index_files[0]))) {
            std::string index_file = index_dir + "/" + index_files[i];
            if (!IsExists(index_file)) {
                ret = false;
            }
        }
        if (ret)
            return true;

        std::string index_files_brute[]
            = { "scann_config.pb", "scann_assets.pbtxt", "dataset.npy" };
        ret = true;
        for (size_t i : Seq(sizeof(index_files_brute) / sizeof(index_files_brute[0]))) {
            std::string index_file = index_dir + "/" + index_files_brute[i];
            if (!IsExists(index_file)) {
                ret = false;
            }
        }
        if (ret)
            is_brute_ = true;
        return ret;
    }

    std::string GetScannConfig(uint64_t datasets_train_size, uint32_t nlist)
    {
        LOG(INFO) << datasets_train_size << ": " << brute_threshold_;
        is_brute_ = false;
        if (datasets_train_size < brute_threshold_) {
            is_brute_ = true;
            LOG(INFO) << "datasets_train_size=" << datasets_train_size;

            std::string ret_config = "num_neighbors: 1 \n\
            distance_measure {distance_measure: \"DotProductDistance\"} \n\
            brute_force { \n\
                fixed_point { \n\
                    enabled: False \n\
                } \n\
            }\n";
            return ret_config;
        }

        std::string config_format = "num_neighbors: %d\n\
            distance_measure {distance_measure: \"DotProductDistance\"} \n\
            partitioning { \n\
                num_children: %d \n\
                min_cluster_size: 50 \n\
                max_clustering_iterations: 12 \n\
                single_machine_center_initialization: RANDOM_INITIALIZATION \n\
                partitioning_distance { \n\
                    distance_measure: \"SquaredL2Distance\" \n\
                } \n\
                query_spilling { \n\
                    spilling_type: FIXED_NUMBER_OF_CENTERS \n\
                    max_spill_centers: %d \n\
                } \n\
                expected_sample_size: %lld \n\
                query_tokenization_distance_override {distance_measure: \"DotProductDistance\"} \n\
                partitioning_type: GENERIC \n\
                query_tokenization_type: FLOAT \n\
            } \n\
            hash { \n\
                asymmetric_hash { \n\
                    lookup_type: INT8_LUT16 \n\
                    use_residual_quantization: True \n\
                    use_global_topn: False \n\
                    quantization_distance { \n\
                        distance_measure: \"SquaredL2Distance\" \n\
                    } \n\
                    num_clusters_per_block: 16 \n\
                    projection { \n\
                        input_dim: %d \n\
                        projection_type: CHUNK \n\
                        num_blocks: %d \n\
                        num_dims_per_block: 2 \n\
                    } \n\
                    noise_shaping_threshold: 0.2 \n\
                    expected_sample_size: %lld \n\
                    max_clustering_iterations: 10 \n\
                } \n\
            } \n\
            exact_reordering { \n\
                approx_num_neighbors: %d \n\
                fixed_point { \n\
                    enabled: True \n\
                } \n\
            } \n";
        int64_t partitioning_expected_sample_size
            = (int64_t)(datasets_train_size * partitioning_train_sample_rate_);
        if (partitioning_expected_sample_size <= nlist) {
            partitioning_expected_sample_size = datasets_train_size;
        }
        int64_t hash_expected_sample_size
            = (int64_t)(datasets_train_size * hash_train_sample_rate_);
        if (hash_expected_sample_size <= nlist) {
            hash_expected_sample_size = datasets_train_size;
        }
        int size_s = std::snprintf(nullptr, 0, config_format.c_str(), (int)topk_, (int)nlist, (int)nprobe_,
            (long long)partitioning_expected_sample_size, (int)dim_, (int)(dim_ / 2),
            (long long)hash_expected_sample_size, (int)reorder_topk_);
        if (size_s <= 0) {
            return std::string();
        }
        auto size = static_cast<size_t>(size_s + 1);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, config_format.c_str(), (int)topk_, (int)nlist, (int)nprobe_,
            (long long)partitioning_expected_sample_size, (int)dim_, (int)(dim_ / 2),
            (long long)hash_expected_sample_size, (int)reorder_topk_);
        return std::string(buf.get(), buf.get() + size - 1);
    }

    bool InitImpl(std::string& index_dir) override;
    int AddImpl(std::vector<float>& datasets, uint64_t npoints, int32_t nthreads) override;
    int TrainImpl(
        std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads) override;
    int DumpImpl(std::string& index_dir, dump_format_t format) override;
    int SearchImpl(const float* queries, int32_t npoints, std::vector<result_search_t>& res,
        std::vector<uint32_t>& counts, params_search_t params) override;

    size_t GetIndexPointsNum() override
    {
        if (scann_handler_ == NULL) {
            return 0;
        }
        return scann_handler_->n_points();
    }

    void ResetImpl() override
    {
        if (scann_handler_ != NULL) {
            delete scann_handler_;
            scann_handler_ = NULL;
        }
    }

    size_t GetPartitioningSize() override
    {
        if (scann_handler_ == NULL) {
            return 0;
        }
        // Number of partitions the index was trained with (0 for brute force).
        const ScannConfig* config = scann_handler_->config();
        if (config == NULL || !config->has_partitioning()) {
            return 0;
        }
        return config->partitioning().num_children();
    }

    virtual bool IsBrute() override { return is_brute_; }

private:
    ScannInterface* scann_handler_;

    uint32_t topk_; // default final_nn
    uint32_t reorder_topk_; // default pre_reorder_nn
    uint32_t nprobe_; // default leaves_to_search
    float partitioning_train_sample_rate_; // default 0.2
    float hash_train_sample_rate_; // default 0.1
    bool is_brute_;
    // Set by CheckIndexFiles: which forms of the AH codes are on disk.
    bool has_packed_hashes_ = false;
    bool has_legacy_hashes_ = false;
};

bool VQLiteIndexScann::InitImpl(std::string& index_dir)
{
    if (!CheckIndexFiles(index_dir)) {
        return true;
    }

    ScannInterface* scann_handler = new ScannInterface();
    if (scann_handler == NULL) {
        return false;
    }

    // The packed codes are used when the directory has them.
    bool use_packed = !is_brute_ && has_packed_hashes_;
    string scann_assets_pbtxt = GetAssetsInfo(index_dir, use_packed);

    ScannConfig config;
    Status ret = ReadProtobufFromFile(string(index_dir) + "/scann_config.pb", &config);
    if (ret.ok() && config.has_hash() && config.hash().has_asymmetric_hash()) {
        // Indexes written by earlier versions have use_global_topn set.  With
        // it this ScaNN scans every leaf through LUT16 GetTopFloatDistances;
        // without it the leaves are scanned with int16 accumulators (what
        // ScaNN 1.2.9 did for batched search), which is markedly faster on
        // large leaves.  It is a search-time switch only: the stored index is
        // the same either way.
        config.mutable_hash()->mutable_asymmetric_hash()->set_use_global_topn(false);
    }
    if (ret.ok()) {
        ret = scann_handler->Initialize(config.DebugString(), scann_assets_pbtxt);
        if (!ret.ok() && use_packed && has_legacy_hashes_) {
            // E.g. a library without packed support dumped into this
            // directory after the packed file was written, which leaves the
            // packed file describing an older state.
            LOG(INFO) << "cannot load " << kPackedLeavesFilename << " (" << ret.message()
                      << "); loading hashed_dataset.npy instead";
            delete scann_handler;
            scann_handler = new ScannInterface();
            scann_assets_pbtxt = GetAssetsInfo(index_dir, false);
            ret = scann_handler->Initialize(config.DebugString(), scann_assets_pbtxt);
        }
    }
    if (!ret.ok()) {
        delete scann_handler;
        LOG(INFO) << ret.message();
        return false;
    }
    // ScannInterface starts a (NumCPUs - 1)-thread pool per index for
    // SearchBatchedParallel, which VQIndex never calls.  Drop it so that every
    // open index does not keep a full set of idle threads.
    scann_handler->SetNumThreads(0);

    this->scann_handler_ = scann_handler;
    LOG(INFO) << "index npoints=" << GetIndexPointsNum();

    return true;
}

int VQLiteIndexScann::AddImpl(std::vector<float>& datasets, uint64_t npoints, int32_t nthreads)
{
    if (this->scann_handler_ == NULL) {
        return -1;
    }
    if (is_brute_) {
        LOG(INFO) << "Add Brute Unavailable";
        return -2;
    }

    // The index is extended through ScaNN's own mutation API rather than by
    // patching its internals: every new point is tokenized with the existing
    // partitioner, its residual is hashed with the existing AH codebook and
    // appended to its leaf, and its int8 copy is appended to the fixed-point
    // reordering dataset.  Centroids and codebooks are left untouched, and new
    // points get consecutive datapoint indices starting at n_points(), which is
    // what keeps them aligned with vids_.
    using Mutator = SingleMachineSearcherBase<float>::Mutator;
    using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;

    if (npoints == 0) {
        return 0;
    }
    if (npoints * dim_ != datasets.size()) {
        LOG(INFO) << "datasets.size[" << datasets.size() << "] != npoints[" << npoints
                  << "] * dim[" << dim_ << "]";
        return -1;
    }

    auto mutator_or = this->scann_handler_->GetMutator();
    if (!mutator_or.ok()) {
        LOG(INFO) << "GetMutator Fail: " << mutator_or.status().message();
        return -1;
    }
    Mutator* mutator = mutator_or.value();

    int32_t num_threads = nthreads;
    int32_t num_cpus = std::max<int32_t>(std::thread::hardware_concurrency(), 1);
    if (num_threads <= 0 || num_threads > num_cpus) {
        num_threads = std::max(num_cpus - 1, 1);
    }
    std::shared_ptr<ThreadPool> pool = StartThreadPool("AddIndex", num_threads);

    const size_t old_npoints = GetIndexPointsNum();
    mutator->Reserve(old_npoints + npoints);

    // Tokenizing + hashing is the expensive part and runs on the pool, one
    // batch at a time; appending to the leaves must be sequential.
    const size_t batch_size = std::max<size_t>(1024, static_cast<size_t>(num_threads) * 256);
    int ret = 0;
    for (size_t begin = 0; begin < npoints && ret == 0; begin += batch_size) {
        const size_t bs = std::min<size_t>(batch_size, npoints - begin);
        std::vector<float> batch_vec(datasets.begin() + begin * dim_,
            datasets.begin() + (begin + bs) * dim_);
        DenseDataset<float> batch(std::move(batch_vec), bs);
        auto precomputed = mutator->ComputePrecomputedMutationArtifacts(batch, pool);

        for (size_t i = 0; i < bs; i++) {
            if (precomputed[i] == nullptr) {
                LOG(INFO) << "Precompute Fail: point=" << begin + i;
                ret = -1;
                break;
            }
            MutationOptions mo;
            mo.precomputed_mutation_artifacts = precomputed[i].get();
            auto idx_or = mutator->AddDatapoint(batch[i], "", mo);
            if (!idx_or.ok()) {
                LOG(INFO) << "AddDatapoint Fail: " << idx_or.status().message();
                ret = -1;
                break;
            }
            if (idx_or.value() != old_npoints + begin + i) {
                LOG(INFO) << "AddDatapoint idx=" << idx_or.value() << "; expected="
                          << old_npoints + begin + i;
                ret = -1;
                break;
            }
        }
    }

    LOG(INFO) << "old_npoints=" << old_npoints << "; add npoints=" << npoints
              << "; index npoints=" << GetIndexPointsNum() << "; ret=" << ret;
    return ret;
}

int VQLiteIndexScann::TrainImpl(
    std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
{
    if (nthreads < 0) {
        return -1;
    }

    if (nlist == 0) {
        nlist = pow(2, ceil(log(sqrt(npoints)) / log(2)));
        if (nlist < 2) {
            nlist = 2;
        }
    }
    // GetScannConfig() switches is_brute_ to what the new index will be; it is
    // put back if the new index does not materialize, because the current
    // index (if any) stays in place then.
    const bool old_is_brute = is_brute_;
    std::string config = GetScannConfig(npoints, nlist);
    if (config.empty()) {
        is_brute_ = old_is_brute;
        return -1;
    }

    std::unique_ptr<ScannInterface> scann_handler(new ScannInterface());

    std::vector<float>*datasets_pre = &datasets, datasets_t;
    if (is_brute_) {
        datasets_t = datasets;
        datasets_pre = &datasets_t;
    }
    int32_t training_threads = nthreads;
    if (training_threads == 0) {
        training_threads = std::max<int32_t>(std::thread::hardware_concurrency(), 1);
    }

    ScannConfig scann_config;
    SingleMachineFactoryOptions opts;
    opts.parallelization_pool = StartThreadPool("scann_threadpool", training_threads - 1);

    // The vector is moved into the DenseDataset (no copy of the float data) and
    // moved back once ScaNN has built the index and no longer references it.
    shared_ptr<DenseDataset<float>> dataset_den
        = std::make_shared<DenseDataset<float>>(std::move(*datasets_pre), npoints);

    Status ret = ParseTextProto(&scann_config, config);
    if (ret.ok()) {
        ret = scann_handler->Initialize(
            std::make_tuple(scann_config, dataset_den, std::move(opts)));
    }

    if (!is_brute_ || !ret.ok()) {
        bool still_needed = false;
        if (ret.ok()) {
            auto needed_or = scann_handler->Float32DatasetIfNeeded();
            still_needed = !needed_or.ok() || needed_or.value() != nullptr;
        }
        if (still_needed) {
            // ScaNN kept a reference to the float dataset; leave it alone and
            // hand the caller a copy instead.
            LOG(INFO) << "float dataset is still referenced by the index, copying it back";
            ConstSpan<float> data = dataset_den->data();
            *datasets_pre = std::vector<float>(data.begin(), data.end());
        } else {
            *datasets_pre = dataset_den->ClearRecyclingDataVector();
        }
    }
    if (!ret.ok()) {
        LOG(INFO) << ret.message();
        is_brute_ = old_is_brute;
        return -1;
    }
    scann_handler->SetNumThreads(0);

    if (!this->is_child_process_ && this->scann_handler_ != NULL) {
        delete this->scann_handler_;
    }
    this->scann_handler_ = scann_handler.release();
    LOG(INFO) << "index npoints=" << GetIndexPointsNum() << "; default_nlist=" << nlist
              << "; is_brute=" << is_brute_;
    return 0;
}

int VQLiteIndexScann::DumpImpl(std::string& index_dir, dump_format_t format)
{
    if (scann_handler_ == NULL) {
        return 0;
    }
    ScannInterface::HashedFormat hashed_format = ScannInterface::kHashedPacked;
    if (format == DUMP_FORMAT_LEGACY) {
        hashed_format = ScannInterface::kHashedNpy;
    } else if (format == DUMP_FORMAT_BOTH) {
        hashed_format = ScannInterface::kHashedBoth;
    }
    StatusOr<ScannAssets> assets_or = scann_handler_->Serialize(index_dir, false, hashed_format);
    if (!assets_or.ok()) {
        LOG(INFO) << assets_or.status().message();
        return -1;
    }
    Status ret = OpenSourceableFileWriter(index_dir + "/scann_assets.pbtxt")
                     .Write(assets_or->DebugString());
    if (!ret.ok()) {
        return -1;
    }
    return 0;
}

int VQLiteIndexScann::SearchImpl(const float* queries, int32_t npoints,
    std::vector<result_search_t>& res, std::vector<uint32_t>& counts, params_search_t params)
{
    if (scann_handler_ == NULL) {
        LOG(INFO) << "scann_handler_ == NULL";
        return -1;
    }

    vector<float> queries_vec(queries, queries + npoints * dim_);
    auto query_dataset = DenseDataset<float>(std::move(queries_vec), npoints);

    std::vector<NNResultsVector> s_res;
    s_res.resize(npoints);
    int32_t topk = topk_;
    if (params.topk_ > 0) {
        topk = params.topk_;
    }
    int32_t reorder_topk = reorder_topk_;
    if (params.reorder_topk_ > 0) {
        reorder_topk = params.reorder_topk_;
    }
    int32_t nprobe = nprobe_;
    if (params.nprobe_ > 0) {
        nprobe = params.nprobe_;
    }
    Status status = scann_handler_->SearchBatched(
        query_dataset, MakeMutableSpan(s_res), topk, reorder_topk, nprobe);
    if (!status.ok()) {
        LOG(INFO) << "search error: " << status.message();
        return -2;
    }

    result_search_t item;
    item.idx_ = 0;
    item.score_ = -1;
    item.vid_ = 0;
    res.assign((size_t)npoints * topk, item);
    counts.assign(npoints, 0);
    for (size_t i : Seq(s_res.size())) {
        size_t n = std::min<size_t>(s_res[i].size(), topk);
        counts[i] = n;
        for (size_t j : Seq(n)) {
            res[i * topk + j].idx_ = s_res[i][j].first;
            res[i * topk + j].score_ = -1 * s_res[i][j].second;
        }
    }
    return 0;
}

}

using namespace vqindex;

void* vqindex_init(const char* index_dir, index_config_t config_i)
{
    if (config_i.index_type_ != INDEX_TYPE_SCANN || config_i.dim_ <= 0) {
        LOG(INFO) << "Only Support ScaNN.";
        return NULL;
    }
    VQLiteIndexScann* vql_index = new VQLiteIndexScann(index_dir, config_i);
    if (vql_index == NULL) {
        LOG(INFO) << "New Molloc Index Fail.";
        return NULL;
    }

    if (!vql_index->Init()) {
        LOG(INFO) << "Init Fail.";
        delete vql_index;
        return NULL;
    }

    return vql_index;
}

void vqindex_release(void* vql_handler)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    delete vql_index;
}

ret_code_t vqindex_dump(void* vql_handler)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Dump();
}

ret_code_t vqindex_dump_format(void* vql_handler, dump_format_t format)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Dump(format);
}

// topk=>final_nn, reorder_topk=>pre_reorder_nn, nprobe=>leaves
ret_code_t vqindex_search(
    void* vql_handler, const float* queries, int len, result_search_t* res, params_search_t params)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }

    std::vector<result_search_t> res_t;
    ret_code_t ret_s = vql_index->Search(queries, len, res_t, params);
    if (ret_s == RET_CODE_OK) {
        memcpy(res, res_t.data(), res_t.size() * sizeof(result_search_t));
    }
    return ret_s;
}

// len: number of datasets float, <npoint = dim_ / len>, <len % dim_ == 0>.
ret_code_t vqindex_add(void* vql_handler, const float* datasets, uint64_t len, const int64_t* vids)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->AddDatasets(datasets, len, vids);
}

ret_code_t vqindex_train(
    void* vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Train(train_type, nlist, nthreads);
}

ret_code_t vqindex_train_process(
    void *vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    VQLiteIndex *vql_index = static_cast<VQLiteIndex *>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->TrainInChildProcess(train_type, nlist, nthreads);
}


index_stats_t vqindex_stats(void* vql_handler)
{
    index_stats_t ret;
    ret.datasets_size_ = 0;
    ret.index_size_ = 0;
    ret.vid_size_ = 0;
    ret.index_nlist_ = 0;
    ret.dim_ = 0;
    ret.is_brute_ = 0;
    ret.brute_threshold_ = 0;
    ret.current_status_ = INDEX_STATE_NONE;

    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return ret;
    }

    vql_index->GetStats(ret);

    return ret;
}
