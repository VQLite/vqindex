/**
 * Copyright 2022 The VQLite Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "scann/scann_ops/cc/vqindex_api.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include <chrono>
#include <thread>

#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/io_oss_wrapper.h"

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
    {
        index_root_dir_ = ".";
        if (index_root_dir != NULL) {
            index_root_dir_ = index_root_dir;
        }
        storage_type_ = STORAGE_FILE;
        if (storage_type_ >= STORAGE_FILE && storage_type_ <= STORAGE_MEMORY) {
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

        ResetImpl();
    }
    virtual int ResetImpl() { LOG(INFO) << "ResetImpl Unavailable"; }

    // add datasets to index, if the index already exists.
    int Add(std::vector<float> &datasets, int32_t nthreads);
    virtual int AddImpl(std::vector<float> &datasets, uint64_t npoints, int32_t nthreads)
    {
        LOG(INFO) << "AddImpl Unavailable";
        return 0;
    }

    ret_code_t Train(train_type_t train_type, uint32_t nlist, int32_t nthreads);
    ret_code_t TrainDefault(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainNew(uint32_t nlist, int32_t nthreads);
    ret_code_t TrainAdd(int32_t nthreads);
    virtual int TrainImpl(std::vector<float> &datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
    {
        LOG(INFO) << "TrainImpl Unavailable";
        return 0;
    }

    ret_code_t Dump();
    virtual int DumpImpl(std::string& index_dir)
    {
        LOG(INFO) << "DumpImpl Unavailable";
        return 0;
    }

    ret_code_t Search(const float* queries, int32_t len, std::vector<result_search_t>& res,
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

protected:
    ret_code_t AddDatasetsMemory(const float* datasets, uint64_t len, const int64_t* vids);
    ret_code_t AddDatasetsFile(const float* datasets, uint64_t len, const int64_t* vids);

    std::string index_root_dir_;
    std::string datasets_filename_;
    std::string vids_filename_;
    std::string index_subdir_name_;

    std::vector<float> datasets_;
    std::vector<int64_t> vids_;

    uint64_t datasets_npoints_;

    uint32_t dim_; // dimensions of vector point
    storage_type_t storage_type_;
    uint64_t brute_threshold_;

    std::mutex mutex_global_lock_;
    std::shared_mutex smutex_vid_rwlock_;
    std::atomic<int> current_search_n_;

    index_state_t current_state_;
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
        if (index_npoints > 0) {
            current_state_ = INDEX_STATE_READY;
        } else {
            current_state_ = INDEX_STATE_NOINDEX;
        }
        
        datasets_npoints_ = datasets_npoints;
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
    if (current_state_ > INDEX_STATE_READY) {
        return RET_CODE_NOREADY;
    }
    if (len % dim_ != 0) {
        LOG(INFO) << "!is_init_ || len % dim_ != 0; current_state_=" << current_state_;
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
    }

    if (ret == RET_CODE_OK) {
        vids_.resize(new_npoints, 0);
        memcpy(vids_.data() + now_npoints, vids, add_npoints * sizeof(int64_t));

        datasets_npoints_ += add_npoints;
    }
    LOG(INFO) << "vids_.size=" << vids_.size() << "; dataset.size=" << datasets_.size() << "; vids_.capacity()=" << vids_.capacity();

    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }

    return ret;
}

int VQLiteIndex::Add(std::vector<float> &datasets, int32_t nthreads)
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

ret_code_t VQLiteIndex::Train(train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
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
    
    return ret;
}

ret_code_t VQLiteIndex::Dump()
{
    std::error_code ec;
    std::string index_dir = index_root_dir_ + "/" + index_subdir_name_;
    ret_code_t ret = RET_CODE_OK;

    if (current_state_ > INDEX_STATE_READY) {
        LOG(INFO) << "current_state_=" << current_state_;
        return RET_CODE_NOREADY;
    }

    std::lock_guard<std::mutex> guard(mutex_global_lock_);

    current_state_ = INDEX_STATE_DUMP;

    if (!IsExists(index_dir) && !std::filesystem::create_directories(index_dir, ec)) {
        LOG(INFO) << "Create Directories Fail.";
        ret = RET_CODE_NOPERMISSION;
        goto end;
    }
    if (DumpImpl(index_dir) != 0) {
        LOG(INFO) << "Dump Index Fail.";
        ret = RET_CODE_INDEXERR;
        goto end;
    }
    if (storage_type_ == STORAGE_MEMORY) {
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
    }

end:
    if (GetIndexPointsNum() > 0) {
        current_state_ = INDEX_STATE_READY;
    } else {
        current_state_ = INDEX_STATE_NOINDEX;
    }
    return ret;
}

ret_code_t VQLiteIndex::Search(
    const float* queries, int32_t len, std::vector<result_search_t>& res, params_search_t params)
{
    if (current_state_ == INDEX_STATE_TRAIN || current_state_ < INDEX_STATE_READY) {
        LOG(INFO) << "current_state_ = INDEX_STATE_TRAIN|INDEX_STATE_NOINDEX";
        return RET_CODE_NOREADY;
    }

    int32_t npoints = len / dim_;
    if (len % dim_ != 0) {
        LOG(INFO) << "Query Len % Dim != 0";
        return RET_CODE_DATAERR;
    }

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

    inline string ReadFileString(const string& filename)
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

    bool CheckIndexFiles(std::string& index_dir)
    {
        std::string index_files[] = { "scann_config.pb", "ah_codebook.pb",
            "serialized_partitioner.pb", "datapoint_to_token.npy", "hashed_dataset.npy",
            "int8_dataset.npy", "int8_multipliers.npy", "dp_norms.npy" };

        bool ret = true;
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

    bool InitImpl(std::string& index_dir) override;
    int AddImpl(std::vector<float> &datasets, uint64_t npoints, int32_t nthreads) override;
    int TrainImpl(
        std::vector<float> &datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads) override;
    int DumpImpl(std::string& index_dir) override;
    int SearchImpl(const float* queries, int32_t npoints, std::vector<result_search_t>& res,
        params_search_t params) override;

    size_t GetIndexPointsNum()
    {
        if (scann_handler_ == NULL) {
            return 0;
        }
        return scann_handler_->n_points();
    }

    int ResetImpl() override
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

private:
    ScannInterface* scann_handler_;

    uint32_t topk_; // default final_nn
    uint32_t reorder_topk_; // default pre_reorder_nn
    uint32_t nprobe_; // default leaves_to_search
    float partitioning_train_sample_rate_; // default 0.2
    float hash_train_sample_rate_; // default 0.1
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

    string scann_assets_pbtxt = ReadFileString(string(index_dir) + "/scann_assets.pbtxt");
    if (scann_assets_pbtxt.empty()) {
        delete scann_handler;
        return false;
    }

    ScannConfig config;
    ReadProtobufFromFile(string(index_dir) + "/scann_config.pb", &config);

    Status ret = scann_handler->Initialize(config.DebugString(), scann_assets_pbtxt);
    if (!ret.ok()) {
        delete scann_handler;
        LOG(INFO) << ret.error_message();
        return false;
    }

    this->scann_handler_ = scann_handler;
    LOG(INFO) << "index npoints=" << GetIndexPointsNum();

    return true;
}

int VQLiteIndexScann::AddImpl(std::vector<float> &datasets, uint64_t npoints, int32_t nthreads)
{
    if (this->scann_handler_ == NULL) {
        return -1;
    }
    if (is_brute_) {
        LOG(INFO) << "Add Brute Unavailable";
        return -2;
    }

    return this->scann_handler_->Add2Index(datasets, npoints, nthreads);
}

int VQLiteIndexScann::TrainImpl(
    std::vector<float> &datasets, uint64_t npoints, uint32_t nlist, int32_t nthreads)
{
    ScannInterface* scann_handler = new ScannInterface();
    if (scann_handler == NULL) {
        return -1;
    }

    if (nlist == 0) {
        nlist = pow(2, ceil(log(sqrt(npoints)) / log(2)));
        if (nlist < 2) {
            nlist = 2;
        }
    }
    std::string config = GetScannConfig(npoints, nlist);
    if (config.empty()) {
        return -1;
    }

    std::vector<float> *datasets_pre = &datasets, datasets_t;
    if (is_brute_) {
        datasets_t = datasets;
        datasets_pre = &datasets_t;
    }
    Status ret = scann_handler->Initialize(*datasets_pre, npoints, config, nthreads, !is_brute_);
    if (!ret.ok()) {
        LOG(INFO) << ret.error_message();
        delete scann_handler;
        return -1;
    }

    if (this->scann_handler_ != NULL) {
        delete this->scann_handler_;
    }
    this->scann_handler_ = scann_handler;
    LOG(INFO) << "index npoints=" << GetIndexPointsNum() << "; default_nlist=" << nlist
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
        LOG(INFO) << assets_or.status().error_message();
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
    std::vector<result_search_t>& res, params_search_t params)
{
    if (scann_handler_ == NULL) {
        LOG(INFO) << "scann_handler_ == NULL";
        return -1;
    }

    vector<float> queries_vec(queries, queries + npoints * dim_);
    auto query_dataset = DenseDataset<float>(queries_vec, npoints);

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
        LOG(INFO) << "search error: " << status.error_message();
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

ret_code_t vqindex_train(void* vql_handler, train_type_t train_type, uint32_t nlist, int32_t nthreads)
{
    VQLiteIndex* vql_index = static_cast<VQLiteIndex*>(vql_handler);
    if (vql_index == NULL) {
        LOG(INFO) << "vql_handler NULL";
        return RET_CODE_NOINIT;
    }
    return vql_index->Train(train_type, nlist, nthreads);
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
