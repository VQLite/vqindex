/**
 * Copyright 2022 The VQLite Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "scann/scann_ops/cc/vqindex_api.h"

#include <stdio.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

#include <chrono>
#include <thread>
#include<sys/wait.h>

#include "scann/scann_ops/cc/scann.h"
#include "scann/proto/auto_tuning.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/io_oss_wrapper.h"

using namespace research_scann;
using namespace std;

namespace vqindex {

thread_local std::string g_last_error;

int64_t MillisecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start)
        .count();
}

ret_code_t CopyStringToBuffer(const std::string& value, char* buffer, uint64_t buffer_len)
{
    if (buffer == NULL || buffer_len == 0) {
        return RET_CODE_MEMORYERR;
    }
    if (value.size() + 1 > buffer_len) {
        return RET_CODE_MEMORYERR;
    }
    memcpy(buffer, value.c_str(), value.size() + 1);
    return RET_CODE_OK;
}

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
        , datasets_dirty_(false)
        , is_child_process_(false)
        , deleted_npoints_(0)
        , last_load_ms_(0)
        , last_dump_ms_(0)
        , last_train_ms_(0)
        , last_rebalance_ms_(0)
        , artifact_format_(ARTIFACT_FORMAT_UNKNOWN)
    {
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

    void SetError(const std::string& error)
    {
        last_error_ = error;
        g_last_error = error;
    }

    void ClearError()
    {
        last_error_.clear();
        g_last_error.clear();
    }

    const std::string& LastError() const { return last_error_; }

    bool IsExists(std::string& file_path) { return std::filesystem::exists(file_path); }

    int64_t GetFileSize(std::string& file_path)
    {
        if (!IsExists(file_path)) {
            return 0;
        }
        return std::filesystem::file_size(file_path);
    }

    void GetStats(index_stats_t& stats)
    {
        stats.datasets_size_ = datasets_npoints_;
        stats.index_size_ = GetIndexPointsNum();
        stats.vid_size_ = vids_.size();
        stats.index_nlist_ = GetPartitioningSize();
        stats.dim_ = dim_;
        if (IsBrute()) {
            stats.is_brute_ = 1;
        }
        stats.brute_threshold_ = brute_threshold_;
        stats.current_status_ = current_state_;
        stats.pending_size_ = stats.vid_size_ - stats.index_size_;
        if (stats.pending_size_ < 0) {
            stats.pending_size_ = 0;
        }
        stats.deleted_size_ = deleted_npoints_;
        stats.last_load_ms_ = last_load_ms_;
        stats.last_dump_ms_ = last_dump_ms_;
        stats.last_train_ms_ = last_train_ms_;
        stats.last_rebalance_ms_ = last_rebalance_ms_;
        stats.artifact_format_ = artifact_format_;
        stats.use_autopilot_ = UseAutopilot() ? 1 : 0;
        stats.enable_soar_ = EnableSoar() ? 1 : 0;
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
    bool LoadAllDatasetsForMutation()
    {
        if (datasets_.size() / dim_ == vids_.size()) {
            return true;
        }
        std::vector<float> datasets;
        if (!LoadTrainDatasets(datasets, 0)) {
            return false;
        }
        if (datasets.size() / dim_ < vids_.size()) {
            return false;
        }
        datasets.resize(vids_.size() * dim_);
        datasets_.swap(datasets);
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
    ret_code_t UpsertDatasets(const float* datasets, uint64_t len, const int64_t* vids);
    ret_code_t DeleteVids(const int64_t* vids, uint64_t n);

    void Reset()
    {
        std::vector<float> t1;
        datasets_.swap(t1);
        std::vector<int64_t> t2;
        vids_.swap(t2);

        datasets_npoints_ = 0;
        datasets_dirty_ = false;

        ResetImpl();
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
    ret_code_t TrainDefault(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainNew(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainAdd(int32_t nthreads);
    virtual int TrainImpl(
        std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
    {
        LOG(INFO) << "TrainImpl Unavailable";
        return 0;
    }
    ret_code_t Rebalance(const char* config_pbtxt, int32_t nthreads);
    virtual int RebalanceImpl(const std::string& config_pbtxt, int32_t nthreads)
    {
        LOG(INFO) << "RebalanceImpl Unavailable";
        return 0;
    }

    ret_code_t Dump();
    ret_code_t DoDump();
    virtual int DumpImpl(std::string& index_dir)
    {
        LOG(INFO) << "DumpImpl Unavailable";
        return 0;
    }

    ret_code_t Search(const float* queries, uint64_t len, std::vector<result_search_t>& res,
        params_search_t params);
    virtual int SearchImpl(const float* queries, int32_t npoints, std::vector<result_search_t>& res,
        params_search_t params)
    {
        LOG(INFO) << "SearchImpl Unavailable";
        return 0;
    }

    uint32_t GetDim() { return dim_; }
    virtual size_t GetPartitioningSize() = 0;
    virtual size_t GetIndexPointsNum() = 0;

    virtual bool IsBrute() { return false; }
    virtual bool UseAutopilot() { return false; }
    virtual bool EnableSoar() { return false; }
    virtual ret_code_t SetTuning(index_tuning_config_t tuning) { return RET_CODE_INDEXERR; }
    virtual ret_code_t SuggestConfig(uint64_t dataset_size, uint32_t nlist, std::string& config)
    {
        return RET_CODE_INDEXERR;
    }
    virtual ret_code_t InitializeHealthStats() { return RET_CODE_INDEXERR; }
    virtual ret_code_t GetHealthStats(index_health_stats_t& stats) { return RET_CODE_INDEXERR; }
    virtual std::string CurrentConfig() { return std::string(); }
    virtual int DeleteImpl(uint64_t idx) { return -1; }
    virtual int UpdateImpl(uint64_t idx, const float* dataset) { return -1; }

protected:
    ret_code_t AddDatasetsMemory(const float* datasets, uint64_t len, const int64_t* vids);
    ret_code_t AddDatasetsFile(const float* datasets, uint64_t len, const int64_t* vids);
    ret_code_t UpsertOne(const float* dataset, int64_t vid);
    bool FindVid(int64_t vid, size_t& idx);
    void RemoveVidAt(size_t idx);
    void UpsertDatasetMemory(size_t idx, const float* dataset);

    std::string index_root_dir_;
    std::string datasets_filename_;
    std::string vids_filename_;
    std::string index_subdir_name_;

    std::vector<float> datasets_;
    std::vector<int64_t> vids_;

    uint64_t datasets_npoints_;
    bool datasets_dirty_;

    uint32_t dim_; // dimensions of vector point
    storage_type_t storage_type_;
    uint64_t brute_threshold_;

    std::mutex mutex_global_lock_;
    std::shared_mutex smutex_vid_rwlock_;
    std::atomic<int> current_search_n_;

    bool is_child_process_;

    index_state_t current_state_;
    uint64_t deleted_npoints_;
    int64_t last_load_ms_;
    int64_t last_dump_ms_;
    int64_t last_train_ms_;
    int64_t last_rebalance_ms_;
    artifact_format_t artifact_format_;
    std::string last_error_;
};

bool VQLiteIndex::Init()
{
    auto start = std::chrono::steady_clock::now();
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
        SetError("init failed");
        Reset();
    } else {
        if (index_npoints > 0) {
            current_state_ = INDEX_STATE_READY;
        } else {
            current_state_ = INDEX_STATE_NOINDEX;
        }

        datasets_npoints_ = datasets_npoints;
    }

    last_load_ms_ = MillisecondsSince(start);
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
    if (current_state_ > INDEX_STATE_READY) {
        SetError("index is busy");
        return RET_CODE_NOREADY;
    }
    if (datasets == NULL || vids == NULL || dim_ == 0 || len % dim_ != 0) {
        LOG(INFO) << "!is_init_ || len % dim_ != 0; current_state_=" << current_state_;
        SetError("invalid dataset length or null input");
        return RET_CODE_DATAERR;
    }

    current_state_ = INDEX_STATE_ADD;

    uint64_t add_npoints = len / dim_;
    uint64_t now_npoints = vids_.size();
    uint64_t new_npoints = now_npoints + add_npoints;

    if (vids_.capacity() < new_npoints) {
        unique_lock<shared_mutex> wulk(smutex_vid_rwlock_);
        size_t new_capacity = vids_.capacity() + add_npoints + 400000;
        vids_.reserve(new_capacity);
        wulk.unlock();
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);

    ret_code_t ret = RET_CODE_OK;
    if (storage_type_ == STORAGE_MEMORY) {
        ret = AddDatasetsMemory(datasets, len, vids);
    } else {
        ret = AddDatasetsFile(datasets, len, vids);
        if (ret == RET_CODE_OK && !datasets_.empty()) {
            ret = AddDatasetsMemory(datasets, len, vids);
            datasets_dirty_ = true;
        }
    }

    if (ret == RET_CODE_OK) {
        vids_.resize(new_npoints, 0);
        memcpy(vids_.data() + now_npoints, vids, add_npoints * sizeof(int64_t));

        datasets_npoints_ += add_npoints;
    }
    LOG(INFO) << "vids_.size=" << vids_.size() << "; dataset.size=" << datasets_.size()
              << "; vids_.capacity()=" << vids_.capacity();

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }

    return ret;
}

bool VQLiteIndex::FindVid(int64_t vid, size_t& idx)
{
    auto it = std::find(vids_.begin(), vids_.end(), vid);
    if (it == vids_.end()) {
        return false;
    }
    idx = static_cast<size_t>(std::distance(vids_.begin(), it));
    return true;
}

void VQLiteIndex::RemoveVidAt(size_t idx)
{
    const size_t old_size = vids_.size();
    if (idx >= old_size) {
        return;
    }
    if (idx + 1 != old_size) {
        vids_[idx] = vids_.back();
        const size_t memory_npoints = datasets_.size() / dim_;
        if (memory_npoints >= old_size) {
            float* dst = datasets_.data() + idx * dim_;
            float* src = datasets_.data() + (old_size - 1) * dim_;
            memcpy(dst, src, dim_ * sizeof(float));
        }
    }
    vids_.pop_back();
    if (datasets_.size() / dim_ >= old_size) {
        datasets_.resize((old_size - 1) * dim_);
        datasets_npoints_ = datasets_.size() / dim_;
        datasets_dirty_ = true;
    }
    deleted_npoints_++;
}

void VQLiteIndex::UpsertDatasetMemory(size_t idx, const float* dataset)
{
    if (dataset == NULL) {
        return;
    }
    const size_t memory_npoints = datasets_.size() / dim_;
    if (idx >= memory_npoints) {
        return;
    }
    memcpy(datasets_.data() + idx * dim_, dataset, dim_ * sizeof(float));
    datasets_dirty_ = true;
}

ret_code_t VQLiteIndex::UpsertOne(const float* dataset, int64_t vid)
{
    size_t idx = 0;
    if (FindVid(vid, idx)) {
        if (GetIndexPointsNum() <= idx) {
            UpsertDatasetMemory(idx, dataset);
            return RET_CODE_OK;
        }
        if (UpdateImpl(idx, dataset) != 0) {
            SetError("update datapoint in index failed");
            return RET_CODE_INDEXERR;
        }
        UpsertDatasetMemory(idx, dataset);
        return RET_CODE_OK;
    }

    ret_code_t ret = RET_CODE_OK;
    if (storage_type_ == STORAGE_MEMORY) {
        ret = AddDatasetsMemory(dataset, dim_, &vid);
    } else {
        ret = AddDatasetsFile(dataset, dim_, &vid);
        if (ret == RET_CODE_OK && !datasets_.empty()) {
            ret = AddDatasetsMemory(dataset, dim_, &vid);
            datasets_dirty_ = true;
        }
    }
    if (ret != RET_CODE_OK) {
        SetError("append upsert datapoint failed");
        return ret;
    }

    {
        unique_lock<shared_mutex> wulk(smutex_vid_rwlock_);
        vids_.push_back(vid);
    }
    datasets_npoints_++;

    if (GetIndexPointsNum() > 0) {
        std::vector<float> add_dataset(dataset, dataset + dim_);
        if (AddImpl(add_dataset, 1, 0) != 0) {
            SetError("add upsert datapoint to index failed");
            return RET_CODE_ADD2INDEXERR;
        }
    }
    return RET_CODE_OK;
}

ret_code_t VQLiteIndex::UpsertDatasets(const float* datasets, uint64_t len, const int64_t* vids)
{
    if (current_state_ > INDEX_STATE_READY) {
        SetError("index is busy");
        return RET_CODE_NOREADY;
    }
    if (datasets == NULL || vids == NULL || dim_ == 0 || len % dim_ != 0) {
        SetError("invalid upsert dataset length or null input");
        return RET_CODE_DATAERR;
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);
    if (storage_type_ == STORAGE_FILE && !LoadAllDatasetsForMutation()) {
        SetError("load file datasets for upsert failed");
        return RET_CODE_DATAERR;
    }
    current_state_ = INDEX_STATE_ADD;
    while (current_search_n_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        LOG(INFO) << "wait search = " << current_search_n_;
    }

    ret_code_t ret = RET_CODE_OK;
    const uint64_t npoints = len / dim_;
    for (uint64_t i = 0; i < npoints; ++i) {
        ret = UpsertOne(datasets + i * dim_, vids[i]);
        if (ret != RET_CODE_OK) {
            break;
        }
    }

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }
    return ret;
}

ret_code_t VQLiteIndex::DeleteVids(const int64_t* vids, uint64_t n)
{
    if (current_state_ > INDEX_STATE_READY || current_state_ < INDEX_STATE_READY) {
        SetError("delete requires a ready index");
        return RET_CODE_NOREADY;
    }
    if (vids == NULL && n > 0) {
        SetError("delete vids is null");
        return RET_CODE_DATAERR;
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);
    if (storage_type_ == STORAGE_FILE && !LoadAllDatasetsForMutation()) {
        SetError("load file datasets for delete failed");
        return RET_CODE_DATAERR;
    }
    current_state_ = INDEX_STATE_ADD;
    while (current_search_n_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        LOG(INFO) << "wait search = " << current_search_n_;
    }

    ret_code_t ret = RET_CODE_OK;
    for (uint64_t i = 0; i < n; ++i) {
        size_t idx = 0;
        while (FindVid(vids[i], idx)) {
            if (idx < GetIndexPointsNum() && DeleteImpl(idx) != 0) {
                SetError("remove datapoint from index failed");
                ret = RET_CODE_INDEXERR;
                break;
            }
            unique_lock<shared_mutex> wulk(smutex_vid_rwlock_);
            RemoveVidAt(idx);
        }
        if (ret != RET_CODE_OK) {
            break;
        }
    }

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }
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

    if (storage_type_ == STORAGE_FILE && datasets_.size() / dim_ != vids_.size()) {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        int64_t datasets_filesize = GetFileSize(datasets_file_path);
        datasets_npoints = datasets_filesize / sizeof(float) / dim_;
    } else {
        datasets_npoints = datasets_.size() / dim_;
    }

    if (datasets_npoints != vids_npoints || datasets_npoints < index_nlist * index_nlist * 2) {
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
    if (storage_type_ == STORAGE_FILE && datasets_.size() / dim_ != vids_.size()) {
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
    if (storage_type_ == STORAGE_FILE && datasets_.size() / dim_ != vids_.size()) {
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

ret_code_t VQLiteIndex::Train(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    auto start = std::chrono::steady_clock::now();
    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
        SetError("index is busy");
        return RET_CODE_NOREADY;
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);

    current_state_ = INDEX_STATE_TRAIN;

    while (current_search_n_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        LOG(INFO) << "wait search = " << current_search_n_;
    }

    ret_code_t ret = RET_CODE_OK;
    switch (train_type) {
    case TRAIN_TYPE_NEW:
        ret = TrainNew(nlist, nthreads);
        break;
    case TRAIN_TYPE_ADD:
        ret = TrainAdd(nthreads);
        break;
    default:
        ret = TrainDefault(nlist, nthreads);
        break;
    }

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }

    last_train_ms_ = MillisecondsSince(start);
    return ret;
}

ret_code_t VQLiteIndex::TrainProcess(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    auto start = std::chrono::steady_clock::now();
    is_child_process_ = true;

    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
        SetError("index is busy");
        return RET_CODE_NOREADY;
    }

    current_state_ = INDEX_STATE_TRAIN;

    ret_code_t ret = RET_CODE_OK;
    switch (train_type) {
    case TRAIN_TYPE_NEW:
        ret = TrainNew(nlist, nthreads);
        break;
    case TRAIN_TYPE_ADD:
        ret = TrainAdd(nthreads);
        break;
    default:
        ret = TrainDefault(nlist, nthreads);
        break;
    }

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }

    last_train_ms_ = MillisecondsSince(start);
    return ret;
}

ret_code_t VQLiteIndex::Rebalance(const char* config_pbtxt, int32_t nthreads)
{
    auto start = std::chrono::steady_clock::now();
    if (current_state_ > INDEX_STATE_READY || current_state_ < INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
        SetError("rebalance requires a ready index");
        return RET_CODE_NOREADY;
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);

    current_state_ = INDEX_STATE_TRAIN;

    while (current_search_n_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        LOG(INFO) << "wait search = " << current_search_n_;
    }

    std::string config;
    if (config_pbtxt != NULL) {
        config = config_pbtxt;
    }

    ret_code_t ret = RET_CODE_OK;
    if (RebalanceImpl(config, nthreads) != 0) {
        ret = RET_CODE_INDEXERR;
    }

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }

    last_rebalance_ms_ = MillisecondsSince(start);
    return ret;
}

ret_code_t VQLiteIndex::DoDump()
{
    auto start = std::chrono::steady_clock::now();
    std::error_code ec;
    std::string index_dir = index_root_dir_ + "/" + index_subdir_name_;
    ret_code_t ret = RET_CODE_OK;

    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
        SetError("index is busy");
        return RET_CODE_NOREADY;
    }

    current_state_ = INDEX_STATE_DUMP;

    if (!IsExists(index_dir) && !std::filesystem::create_directories(index_dir, ec)) {
        LOG(INFO) << "Create Directories Fail.";
        ret = RET_CODE_NOPERMISSION;
        goto end;
    }
    if (DumpImpl(index_dir) != 0) {
        LOG(INFO) << "Dump Index Fail.";
        SetError("dump index failed");
        ret = RET_CODE_INDEXERR;
        goto end;
    }
    {
        std::string vids_file_path = index_root_dir_ + "/" + vids_filename_;
        std::ofstream outs_vids(vids_file_path, ios::out | ios::binary);
        if (!outs_vids) {
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
        outs_vids.write((char*)vids_.data(), vids_.size() * sizeof(int64_t));
        outs_vids.close();
        if (outs_vids.bad()) {
            LOG(INFO) << "Dump vids Fail.";
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
    }
    if (storage_type_ == STORAGE_MEMORY || datasets_dirty_) {
        std::string datasets_file_path = index_root_dir_ + "/" + datasets_filename_;
        std::ofstream outs_datasets(datasets_file_path, ios::out | ios::binary);
        if (!outs_datasets) {
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
        outs_datasets.write((char*)datasets_.data(), datasets_.size() * sizeof(float));
        outs_datasets.close();
        if (outs_datasets.bad()) {
            LOG(INFO) << "Dump Datasets Fail.";
            ret = RET_CODE_NOPERMISSION;
            goto end;
        }
        datasets_npoints_ = datasets_.size() / dim_;
        datasets_dirty_ = false;
    }

end:
    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }
    last_dump_ms_ = MillisecondsSince(start);
    return ret;
}

ret_code_t VQLiteIndex::Dump()
{
    if (this->is_child_process_) {
        return this->DoDump();
    }
    std::lock_guard<std::mutex> guard(mutex_global_lock_);
    return this->DoDump();
}

ret_code_t VQLiteIndex::Search(
    const float* queries, uint64_t len, std::vector<result_search_t>& res, params_search_t params)
{
    if (current_state_ == INDEX_STATE_TRAIN || current_state_ < INDEX_STATE_READY) {
        LOG(INFO) << "current_state_ = INDEX_STATE_TRAIN|INDEX_STATE_NOINDEX";
        SetError("search requires a ready index");
        return RET_CODE_NOREADY;
    }

    if (queries == NULL || dim_ == 0 || len % dim_ != 0) {
        LOG(INFO) << "Query Len % Dim != 0";
        SetError("invalid query length or null input");
        return RET_CODE_DATAERR;
    }
    uint64_t npoints_u64 = len / dim_;
    if (npoints_u64 > INT32_MAX) {
        SetError("too many query points");
        return RET_CODE_DATAERR;
    }
    int32_t npoints = static_cast<int32_t>(npoints_u64);

    current_search_n_++;
    int ret_s = SearchImpl(queries, npoints, res, params);
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

    shared_lock<shared_mutex> rslk(smutex_vid_rwlock_);
    for (size_t i : Seq(res.size())) {
        if (res[i].score_ >= 0) {
            res[i].vid_ = vids_[res[i].idx_];
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
        , use_autopilot_(false)
        , enable_soar_(false)
        , autopilot_reordering_dtype_(AutopilotTreeAH::INT8)
        , soar_lambda_(1.5f)
        , soar_overretrieve_factor_(2.0f)
        , autopilot_l1_size_(32768)
        , autopilot_l3_size_(33554432)
        , is_brute_(false)
    {
        if (partitioning_train_sample_rate_ < 0 || partitioning_train_sample_rate_ > 1) {
            partitioning_train_sample_rate_ = 0.2;
        }
        if (hash_train_sample_rate_ < 0 || hash_train_sample_rate_ > 1) {
            hash_train_sample_rate_ = 0.1;
        }
    }

    uint32_t DefaultNlist(uint64_t npoints)
    {
        uint32_t nlist = pow(2, ceil(log(sqrt(npoints)) / log(2)));
        if (nlist < 2) {
            nlist = 2;
        }
        return nlist;
    }

    ret_code_t SetTuning(index_tuning_config_t tuning) override
    {
        if (tuning.topk_ > 0) {
            topk_ = tuning.topk_;
        }
        if (tuning.reorder_topk_ > 0) {
            reorder_topk_ = tuning.reorder_topk_;
        }
        if (tuning.nprobe_ > 0) {
            nprobe_ = tuning.nprobe_;
        }
        use_autopilot_ = tuning.use_autopilot_ != 0;
        enable_soar_ = tuning.enable_soar_ != 0;
        if (tuning.autopilot_reordering_dtype_ == AutopilotTreeAH::BFLOAT16 ||
            tuning.autopilot_reordering_dtype_ == AutopilotTreeAH::INT8 ||
            tuning.autopilot_reordering_dtype_ == AutopilotTreeAH::FLOAT32) {
            autopilot_reordering_dtype_ = tuning.autopilot_reordering_dtype_;
        }
        if (tuning.soar_lambda_ > 0.0f) {
            soar_lambda_ = tuning.soar_lambda_;
        }
        if (tuning.soar_overretrieve_factor_ >= 1.0f &&
            tuning.soar_overretrieve_factor_ <= 2.0f) {
            soar_overretrieve_factor_ = tuning.soar_overretrieve_factor_;
        }
        if (tuning.autopilot_l1_size_ > 0) {
            autopilot_l1_size_ = tuning.autopilot_l1_size_;
        }
        if (tuning.autopilot_l3_size_ > 0) {
            autopilot_l3_size_ = tuning.autopilot_l3_size_;
        }
        return RET_CODE_OK;
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

    bool AssetExists(std::string& index_dir, const std::string& filename)
    {
        std::string file_path = index_dir + "/" + filename;
        return IsExists(file_path);
    }

    std::string AssetInfo(const std::string& asset_type, const std::string& asset_path)
    {
        return "assets { \n\
                asset_type: " + asset_type + " \n\
                asset_path: \"" + asset_path + "\" \n\
            } \n";
    }

    std::string GetAssetsInfo(std::string& index_dir)
    {
        std::string assets_info;
        assets_info += AssetInfo("AH_CENTERS", index_dir + "/ah_codebook.pb");
        assets_info += AssetInfo("PARTITIONER", index_dir + "/serialized_partitioner.pb");
        assets_info += AssetInfo("TOKENIZATION_NPY", index_dir + "/datapoint_to_token.npy");

        const bool has_packed_lut16 =
            AssetExists(index_dir, "leaf_lut16_packed_dataset.npy") &&
            AssetExists(index_dir, "leaf_lut16_packed_meta.npy");
        if (has_packed_lut16) {
            assets_info += AssetInfo("AH_LEAF_LUT16_PACKED_DATASET_NPY",
                index_dir + "/leaf_lut16_packed_dataset.npy");
            assets_info += AssetInfo("AH_LEAF_LUT16_PACKED_META_NPY",
                index_dir + "/leaf_lut16_packed_meta.npy");
        } else {
            assets_info += AssetInfo("AH_DATASET_NPY", index_dir + "/hashed_dataset.npy");
            if (AssetExists(index_dir, "hashed_dataset_soar.npy")) {
                assets_info +=
                    AssetInfo("AH_DATASET_SOAR_NPY", index_dir + "/hashed_dataset_soar.npy");
            }
        }
        assets_info += AssetInfo("INT8_DATASET_NPY", index_dir + "/int8_dataset.npy");
        assets_info += AssetInfo("INT8_MULTIPLIERS_NPY", index_dir + "/int8_multipliers.npy");
        assets_info += AssetInfo("INT8_NORMS_NPY", index_dir + "/dp_norms.npy");

        std::string assets_info_brute = "assets { \n\
                asset_type: DATASET_NPY \n\
                asset_path: \"$index_dir/dataset.npy\" \n\
            }";

        if (is_brute_) {
            return std::regex_replace(assets_info_brute, std::regex("\\$index_dir"), index_dir);
        }
        return assets_info;
    }

    bool CheckIndexFiles(std::string& index_dir)
    {
        std::string packed_index_files[] = { "scann_config.pb", "ah_codebook.pb",
            "serialized_partitioner.pb", "datapoint_to_token.npy",
            "leaf_lut16_packed_dataset.npy", "leaf_lut16_packed_meta.npy",
            "int8_dataset.npy", "int8_multipliers.npy", "dp_norms.npy" };

        bool ret = true;
        for (size_t i : Seq(sizeof(packed_index_files) / sizeof(packed_index_files[0]))) {
            std::string index_file = index_dir + "/" + packed_index_files[i];
            if (!IsExists(index_file)) {
                ret = false;
            }
        }
        if (ret) {
            artifact_format_ = ARTIFACT_FORMAT_LEAF_LUT16_PACKED;
            return true;
        }

        std::string legacy_index_files[] = { "scann_config.pb", "ah_codebook.pb",
            "serialized_partitioner.pb", "datapoint_to_token.npy",
            "hashed_dataset.npy", "int8_dataset.npy", "int8_multipliers.npy", "dp_norms.npy" };
        ret = true;
        for (size_t i : Seq(sizeof(legacy_index_files) / sizeof(legacy_index_files[0]))) {
            std::string index_file = index_dir + "/" + legacy_index_files[i];
            if (!IsExists(index_file)) {
                ret = false;
            }
        }
        if (ret) {
            artifact_format_ = ARTIFACT_FORMAT_LEGACY_HASHED;
            return true;
        }

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
        if (ret)
            artifact_format_ = ARTIFACT_FORMAT_BRUTE;
        return ret;
    }

    std::string GetManualScannConfig(uint64_t datasets_train_size, uint32_t nlist)
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
                expected_sample_size: %d \n\
                query_tokenization_distance_override {distance_measure: \"DotProductDistance\"} \n\
                partitioning_type: GENERIC \n\
                query_tokenization_type: FLOAT \n\
            } \n\
            hash { \n\
                asymmetric_hash { \n\
                    lookup_type: INT8_LUT16 \n\
                    use_residual_quantization: True \n\
                    use_global_topn: True \n\
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
                    expected_sample_size: %d \n\
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
        int size_s = std::snprintf(nullptr, 0, config_format.c_str(), topk_, nlist, nprobe_,
            partitioning_expected_sample_size, dim_, dim_ / 2, hash_expected_sample_size,
            reorder_topk_);
        if (size_s <= 0) {
            return std::string();
        }
        auto size = static_cast<size_t>(size_s + 1);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, config_format.c_str(), topk_, nlist, nprobe_,
            partitioning_expected_sample_size, dim_, dim_ / 2, hash_expected_sample_size,
            reorder_topk_);
        return std::string(buf.get(), buf.get() + size - 1);
    }

    bool ApplySoarToConfig(ScannConfig& config)
    {
        if (!enable_soar_ || !config.has_partitioning()) {
            return true;
        }
        auto* database_spilling =
            config.mutable_partitioning()->mutable_database_spilling();
        database_spilling->set_spilling_type(DatabaseSpillingConfig::SOAR);
        database_spilling->set_orthogonality_amplification_lambda(soar_lambda_);
        database_spilling->set_overretrieve_factor(soar_overretrieve_factor_);
        return true;
    }

    std::string GetAutopilotScannConfig(uint64_t datasets_train_size, uint32_t nlist)
    {
        is_brute_ = false;
        if (datasets_train_size < brute_threshold_) {
            return GetManualScannConfig(datasets_train_size, nlist);
        }

        ScannConfig config;
        config.set_num_neighbors(topk_);
        config.mutable_distance_measure()->set_distance_measure("DotProductDistance");
        auto* tree_ah = config.mutable_autopilot()->mutable_tree_ah();
        tree_ah->set_reordering_dtype(
            static_cast<AutopilotTreeAH::DataType>(autopilot_reordering_dtype_));
        tree_ah->set_l1_size(autopilot_l1_size_);
        tree_ah->set_l3_size(autopilot_l3_size_);
        if (nlist > 0) {
            tree_ah->set_num_leaf_partitions(nlist);
        }

        StatusOr<ScannConfig> tuned =
            Autopilot(config, nullptr, datasets_train_size, dim_);
        if (!tuned.ok()) {
            LOG(INFO) << "Autopilot failed: " << tuned.status().message();
            return std::string();
        }
        ApplySoarToConfig(*tuned);
        tuned->clear_autopilot();

        std::string ret_config;
        if (!google::protobuf::TextFormat::PrintToString(*tuned, &ret_config)) {
            return std::string();
        }
        return ret_config;
    }

    std::string GetScannConfig(uint64_t datasets_train_size, uint32_t nlist)
    {
        std::string config_text = use_autopilot_
            ? GetAutopilotScannConfig(datasets_train_size, nlist)
            : GetManualScannConfig(datasets_train_size, nlist);
        if (config_text.empty() || !enable_soar_) {
            return config_text;
        }

        ScannConfig config;
        if (!google::protobuf::TextFormat::ParseFromString(config_text, &config)) {
            return std::string();
        }
        ApplySoarToConfig(config);
        std::string ret_config;
        if (!google::protobuf::TextFormat::PrintToString(config, &ret_config)) {
            return std::string();
        }
        return ret_config;
    }

    ret_code_t SuggestConfig(uint64_t dataset_size, uint32_t nlist, std::string& config) override
    {
        if (dataset_size == 0) {
            dataset_size = datasets_npoints_;
        }
        if (dataset_size == 0) {
            return RET_CODE_DATAERR;
        }
        if (nlist == 0) {
            nlist = DefaultNlist(dataset_size);
        }
        config = GetScannConfig(dataset_size, nlist);
        return config.empty() ? RET_CODE_INDEXERR : RET_CODE_OK;
    }

    bool InitImpl(std::string& index_dir) override;
    int AddImpl(std::vector<float>& datasets, uint64_t npoints, int32_t nthreads) override;
    int TrainImpl(
        std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads) override;
    int RebalanceImpl(const std::string& config_pbtxt, int32_t nthreads) override;
    int DumpImpl(std::string& index_dir) override;
    int SearchImpl(const float* queries, int32_t npoints, std::vector<result_search_t>& res,
        params_search_t params) override;
    ret_code_t InitializeHealthStats() override;
    ret_code_t GetHealthStats(index_health_stats_t& stats) override;
    std::string CurrentConfig() override;
    int DeleteImpl(uint64_t idx) override;
    int UpdateImpl(uint64_t idx, const float* dataset) override;

    size_t GetIndexPointsNum()
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
        return scann_handler_->GetPartitioningSize();
    }

    virtual bool IsBrute() override { return is_brute_; }
    bool UseAutopilot() override { return use_autopilot_; }
    bool EnableSoar() override { return enable_soar_; }

private:
    ScannInterface* scann_handler_;

    uint32_t topk_; // default final_nn
    uint32_t reorder_topk_; // default pre_reorder_nn
    uint32_t nprobe_; // default leaves_to_search
    float partitioning_train_sample_rate_; // default 0.2
    float hash_train_sample_rate_; // default 0.1
    bool use_autopilot_;
    bool enable_soar_;
    uint32_t autopilot_reordering_dtype_;
    float soar_lambda_;
    float soar_overretrieve_factor_;
    uint64_t autopilot_l1_size_;
    uint64_t autopilot_l3_size_;
    bool is_brute_;
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

    string scann_assets_pbtxt = GetAssetsInfo(index_dir);

    ScannConfig config;
    ReadProtobufFromFile(string(index_dir) + "/scann_config.pb", &config);

    Status ret = scann_handler->Initialize(config.DebugString(), scann_assets_pbtxt);
    if (!ret.ok()) {
        delete scann_handler;
        LOG(INFO) << ret.message();
        return false;
    }

    this->scann_handler_ = scann_handler;
    LOG(INFO) << "index npoints=" << GetIndexPointsNum();

    return true;
}

int VQLiteIndexScann::AddImpl(std::vector<float>& datasets, uint64_t npoints, int32_t nthreads)
{
    if (this->scann_handler_ == NULL) {
        return -1;
    }

    return this->scann_handler_->Add2Index(datasets, npoints, nthreads);
}

int VQLiteIndexScann::TrainImpl(
    std::vector<float>& datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
{
    ScannInterface* scann_handler = new ScannInterface();
    if (scann_handler == NULL) {
        return -1;
    }

    if (nlist == 0) {
        nlist = DefaultNlist(npoints);
    }
    std::string config = GetScannConfig(npoints, nlist);
    if (config.empty()) {
        delete scann_handler;
        return -1;
    }

    std::vector<float>*datasets_pre = &datasets, datasets_t;
    if (is_brute_) {
        datasets_t = datasets;
        datasets_pre = &datasets_t;
    }
    Status ret = scann_handler->Initialize(*datasets_pre, npoints, config, nthreads, !is_brute_);
    if (!ret.ok()) {
        LOG(INFO) << ret.message();
        delete scann_handler;
        return -1;
    }

    if (!this->is_child_process_ && this->scann_handler_ != NULL) {
        delete this->scann_handler_;
    }
    this->scann_handler_ = scann_handler;
    artifact_format_ = is_brute_ ? ARTIFACT_FORMAT_BRUTE : ARTIFACT_FORMAT_LEAF_LUT16_PACKED;
    LOG(INFO) << "index npoints=" << GetIndexPointsNum() << "; default_nlist=" << nlist
              << "; is_brute=" << is_brute_;
    return 0;
}

int VQLiteIndexScann::RebalanceImpl(const std::string& config_pbtxt, int32_t nthreads)
{
    if (this->scann_handler_ == NULL || is_brute_) {
        return -1;
    }
    if (nthreads > 0) {
        this->scann_handler_->SetNumThreads(nthreads);
    }

    std::string config = config_pbtxt;
    if (config.empty()) {
        uint32_t nlist = GetPartitioningSize();
        if (nlist == 0) {
            nlist = DefaultNlist(datasets_npoints_);
        }
        config = GetScannConfig(datasets_npoints_, nlist);
    }
    if (config.empty()) {
        return -1;
    }

    StatusOr<ScannConfig> ret = this->scann_handler_->RetrainAndReindex(config);
    if (!ret.ok()) {
        LOG(INFO) << "rebalance error: " << ret.status().message();
        return -1;
    }
    is_brute_ = ret->has_brute_force();
    artifact_format_ = is_brute_ ? ARTIFACT_FORMAT_BRUTE : ARTIFACT_FORMAT_LEAF_LUT16_PACKED;
    LOG(INFO) << "rebalance done; index npoints=" << GetIndexPointsNum()
              << "; is_brute=" << is_brute_;
    return 0;
}

int VQLiteIndexScann::DumpImpl(std::string& index_dir)
{
    if (scann_handler_ == NULL) {
        return 0;
    }
    StatusOr<ScannAssets> assets_or = scann_handler_->Serialize(index_dir);
    if (!assets_or.ok()) {
        LOG(INFO) << assets_or.status().message();
        return -1;
    }
    Status ret = OpenSourceableFileWriter(index_dir + "/scann_assets.pbtxt")
                     .Write(assets_or->DebugString());
    if (!ret.ok()) {
        return -1;
    }
    bool wrote_packed_lut16 = false;
    for (const auto& asset : assets_or->assets()) {
        if (asset.asset_type() == ScannAsset::AH_LEAF_LUT16_PACKED_DATASET_NPY ||
            asset.asset_type() == ScannAsset::AH_LEAF_LUT16_PACKED_META_NPY) {
            wrote_packed_lut16 = true;
        }
    }
    if (wrote_packed_lut16) {
        artifact_format_ = ARTIFACT_FORMAT_LEAF_LUT16_PACKED;
        std::error_code ec;
        std::filesystem::remove(index_dir + "/hashed_dataset.npy", ec);
        ec.clear();
        std::filesystem::remove(index_dir + "/hashed_dataset_soar.npy", ec);
    } else if (is_brute_) {
        artifact_format_ = ARTIFACT_FORMAT_BRUTE;
    } else {
        artifact_format_ = ARTIFACT_FORMAT_LEGACY_HASHED;
    }
    return 0;
}

std::string VQLiteIndexScann::CurrentConfig()
{
    if (scann_handler_ == NULL) {
        return std::string();
    }
    return scann_handler_->CurrentConfig();
}

int VQLiteIndexScann::DeleteImpl(uint64_t idx)
{
    if (scann_handler_ == NULL) {
        return -1;
    }
    return scann_handler_->RemoveFromIndex(static_cast<DatapointIndex>(idx));
}

int VQLiteIndexScann::UpdateImpl(uint64_t idx, const float* dataset)
{
    if (scann_handler_ == NULL) {
        return -1;
    }
    return scann_handler_->UpdateInIndex(static_cast<DatapointIndex>(idx), dataset);
}

ret_code_t VQLiteIndexScann::InitializeHealthStats()
{
    if (scann_handler_ == NULL) {
        return RET_CODE_NOREADY;
    }
    Status ret = scann_handler_->InitializeHealthStats();
    if (!ret.ok()) {
        LOG(INFO) << "InitializeHealthStats error: " << ret.message();
        return RET_CODE_INDEXERR;
    }
    return RET_CODE_OK;
}

ret_code_t VQLiteIndexScann::GetHealthStats(index_health_stats_t& stats)
{
    if (scann_handler_ == NULL) {
        return RET_CODE_NOREADY;
    }
    StatusOr<ScannInterface::ScannHealthStats> ret = scann_handler_->GetHealthStats();
    if (!ret.ok()) {
        LOG(INFO) << "GetHealthStats error: " << ret.status().message();
        return RET_CODE_INDEXERR;
    }
    stats.partition_weighted_avg_relative_imbalance_ =
        ret->partition_weighted_avg_relative_imbalance;
    stats.partition_avg_relative_positive_imbalance_ =
        ret->partition_avg_relative_positive_imbalance;
    stats.avg_quantization_error_ = ret->avg_quantization_error;
    stats.sum_partition_sizes_ = ret->sum_partition_sizes;
    return RET_CODE_OK;
}

int VQLiteIndexScann::SearchImpl(const float* queries, int32_t npoints,
    std::vector<result_search_t>& res, params_search_t params)
{
    if (scann_handler_ == NULL) {
        LOG(INFO) << "scann_handler_ == NULL";
        return -1;
    }

    vector<float> queries_vec(queries, queries + npoints * dim_);
    auto query_dataset = DenseDataset<float>(std::move(queries_vec), npoints);

    std::vector<std::vector<std::pair<uint32_t, float>>> s_res;
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
    res.resize(npoints * topk, item);
    for (size_t i : Seq(s_res.size())) {
        for (size_t j : Seq(s_res[i].size())) {
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
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Dump();
}

ret_code_t vqindex_flush(void* vql_handler)
{
    return vqindex_dump(vql_handler);
}

// topk=>final_nn, reorder_topk=>pre_reorder_nn, nprobe=>leaves
ret_code_t vqindex_search(
    void* vql_handler, const float* queries, uint64_t len, result_search_t* res,
    uint64_t res_capacity, uint64_t* res_count, params_search_t params)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    if (res_count == NULL) {
        vql_index->SetError("res_count is null");
        return RET_CODE_MEMORYERR;
    }
    *res_count = 0;
    if (res == NULL && res_capacity > 0) {
        vql_index->SetError("result buffer is null");
        return RET_CODE_MEMORYERR;
    }

    std::vector<result_search_t> res_t;
    ret_code_t ret_s = vql_index->Search(queries, len, res_t, params);
    if (ret_s == RET_CODE_OK) {
        *res_count = res_t.size();
        if (res_t.size() > res_capacity) {
            vql_index->SetError("result buffer capacity is too small");
            return RET_CODE_MEMORYERR;
        }
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
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->AddDatasets(datasets, len, vids);
}

ret_code_t vqindex_insert(void* vql_handler, const float* datasets, uint64_t len, const int64_t* vids)
{
    return vqindex_add(vql_handler, datasets, len, vids);
}

ret_code_t vqindex_upsert(void* vql_handler, const float* datasets, uint64_t len, const int64_t* vids)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->UpsertDatasets(datasets, len, vids);
}

ret_code_t vqindex_delete(void* vql_handler, const int64_t* vids, uint64_t n)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->DeleteVids(vids, n);
}

ret_code_t vqindex_train(
    void* vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Train(train_type, nlist, nthreads);
}

ret_code_t vqindex_train_process(
    void *vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    ret_code_t ret_code = RET_CODE_OK;
    int child_status = 0;
    pid_t pid = 0;

    VQLiteIndex *vql_index = static_cast<VQLiteIndex *>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }

    pid = fork(); 
    if (pid == -1) {
        LOG(INFO) << "fork error.";
        return RET_CODE_ERR;
    } else if (pid == 0) {
        LOG(INFO) << "in child process, pid=" << pid;
        ret_code_t cret_code = vql_index->TrainProcess(train_type, nlist, nthreads);
        if (cret_code == RET_CODE_OK) {
            LOG(INFO) << "train suss, in dump.";
            cret_code = vql_index->Dump();
            LOG(INFO) << "out dump, cret_code=" << cret_code;
        }
        LOG(INFO) << "out child process, pid=" << pid << "; cret_code=" << cret_code;
        exit(cret_code == RET_CODE_OK ? 0 : -cret_code);
    } else {
        LOG(INFO) << "in parent process, waitting.";
        wait(&child_status);
        if (WIFEXITED(child_status)) {
            int exit_code = WEXITSTATUS(child_status);
            ret_code = exit_code == 0 ? RET_CODE_OK : (ret_code_t)(-exit_code);
        } else {
            ret_code = RET_CODE_ERR;
        }
        LOG(INFO) << "out parent process, ret_code=" << ret_code << "; child_status" << child_status;
    }

    return ret_code;
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
    ret.pending_size_ = 0;
    ret.deleted_size_ = 0;
    ret.last_load_ms_ = 0;
    ret.last_dump_ms_ = 0;
    ret.last_train_ms_ = 0;
    ret.last_rebalance_ms_ = 0;
    ret.artifact_format_ = ARTIFACT_FORMAT_UNKNOWN;
    ret.use_autopilot_ = 0;
    ret.enable_soar_ = 0;

    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return ret;
    }

    vql_index->GetStats(ret);

    return ret;
}

ret_code_t vqindex_last_error(void* vql_handler, char* error_msg, uint64_t error_msg_len)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    const std::string& error = vql_index == NULL ? g_last_error : vql_index->LastError();
    return CopyStringToBuffer(error, error_msg, error_msg_len);
}

ret_code_t vqindex_clear_error(void* vql_handler)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        g_last_error.clear();
        return RET_CODE_OK;
    }
    vql_index->ClearError();
    return RET_CODE_OK;
}

ret_code_t vqindex_version(char* version, uint64_t version_len)
{
    return CopyStringToBuffer(
        "vqindex_api=2;scann=google-research-current;artifact=leaf_lut16_packed",
        version, version_len);
}

ret_code_t vqindex_capabilities(char* capabilities, uint64_t capabilities_len)
{
    return CopyStringToBuffer(
        "packed_lut16=1;legacy_hashed_load=1;flush=1;insert=1;upsert=1;delete=1;"
        "last_error=1;current_config=1;autopilot=1;soar=1;health_stats=1",
        capabilities, capabilities_len);
}

ret_code_t vqindex_set_tuning(void* vql_handler, index_tuning_config_t tuning)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->SetTuning(tuning);
}

ret_code_t vqindex_suggest_config(
    void* vql_handler, uint64_t dataset_size, uint32_t nlist,
    char* config_pbtxt, uint64_t config_pbtxt_len)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    if (config_pbtxt == NULL || config_pbtxt_len == 0) {
        return RET_CODE_MEMORYERR;
    }

    std::string config;
    ret_code_t ret = vql_index->SuggestConfig(dataset_size, nlist, config);
    if (ret != RET_CODE_OK) {
        return ret;
    }
    if (config.size() + 1 > config_pbtxt_len) {
        return RET_CODE_MEMORYERR;
    }
    memcpy(config_pbtxt, config.c_str(), config.size() + 1);
    return RET_CODE_OK;
}

ret_code_t vqindex_current_config(
    void* vql_handler, char* config_pbtxt, uint64_t config_pbtxt_len)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        g_last_error = "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return CopyStringToBuffer(vql_index->CurrentConfig(), config_pbtxt, config_pbtxt_len);
}

ret_code_t vqindex_rebalance(void* vql_handler, const char* config_pbtxt, int32_t nthreads)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Rebalance(config_pbtxt, nthreads);
}

ret_code_t vqindex_initialize_health_stats(void* vql_handler)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->InitializeHealthStats();
}

ret_code_t vqindex_health_stats(void* vql_handler, index_health_stats_t* stats)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    if (stats == NULL) {
        return RET_CODE_MEMORYERR;
    }
    stats->partition_weighted_avg_relative_imbalance_ = 0;
    stats->partition_avg_relative_positive_imbalance_ = 0;
    stats->avg_quantization_error_ = 0;
    stats->sum_partition_sizes_ = 0;
    return vql_index->GetHealthStats(*stats);
}
