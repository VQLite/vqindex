// Copyright 2026 The VQLite Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// On-disk container for the asymmetric-hashing codes of a tree-AH index in the
// layout the LUT16 scan uses in memory: one bit-packed block per partition.
//
// hashed_dataset.npy stores one byte per code in datapoint order, so loading
// it means holding that whole array while every partition is regrouped and
// packed.  This file stores each partition's packed bytes as they sit in
// asymmetric_hashing2::PackedDataset::bit_packed_data, so a partition is read
// straight into its final buffer.
//
// Layout (little endian):
//   header, 64 bytes:
//     char     magic[8]      "VQAHPK16"
//     uint32   version       1
//     uint32   num_blocks    codes per datapoint
//     uint32   num_leaves
//     uint32   reserved      0
//     uint64   num_datapoints
//     uint64   tokenization_fingerprint
//     uint64   data_offset   start of leaf 0's bytes, a multiple of 64
//     uint64   reserved[2]   0
//   uint32 leaf_sizes[num_leaves]        datapoints in each partition
//   zero padding up to data_offset
//   leaf 0 bytes, leaf 1 bytes, ...      PackedLeafBytes(leaf_size, num_blocks)
//
// Within a partition, datapoints are in ascending datapoint-index order, which
// is the order datapoint_to_token.npy yields when it is loaded.  The
// fingerprint ties the file to that tokenization: it differs whenever a
// datapoint was added or moved to another partition, so codes written for one
// state of the index are never applied to another.

#ifndef SCANN_UTILS_PACKED_LEAVES_FILE_H_
#define SCANN_UTILS_PACKED_LEAVES_FILE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

inline constexpr char kPackedLeavesFilename[] = "hashed_dataset_packed.bin";

inline size_t PackedLeafBytes(uint64_t num_datapoints, uint64_t num_blocks) {
  return num_blocks * ((num_datapoints + 31) & ~uint64_t{31}) / 2;
}

uint64_t TokenizationFingerprint(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

uint64_t TokenizationFingerprint(ConstSpan<int32_t> datapoint_to_token,
                                 size_t num_leaves);

class PackedLeavesReader {
 public:
  static StatusOr<std::unique_ptr<PackedLeavesReader>> Open(
      const std::string& path);
  ~PackedLeavesReader();

  PackedLeavesReader(const PackedLeavesReader&) = delete;
  PackedLeavesReader& operator=(const PackedLeavesReader&) = delete;

  uint32_t num_blocks() const { return num_blocks_; }
  size_t num_leaves() const { return leaf_sizes_.size(); }
  uint64_t num_datapoints() const { return num_datapoints_; }
  uint64_t tokenization_fingerprint() const { return fingerprint_; }
  uint32_t leaf_size(size_t leaf) const { return leaf_sizes_[leaf]; }
  const std::string& path() const { return path_; }

  // Reads one partition's packed bytes into `out` (resized to fit).  Safe to
  // call from several threads at once.
  Status ReadLeaf(size_t leaf, std::vector<uint8_t>* out) const;

 private:
  PackedLeavesReader() = default;

  std::string path_;
  int fd_ = -1;
  uint32_t num_blocks_ = 0;
  uint64_t num_datapoints_ = 0;
  uint64_t fingerprint_ = 0;
  std::vector<uint32_t> leaf_sizes_;
  std::vector<uint64_t> leaf_offsets_;
};

// Writes to "<path>.tmp" and renames onto `path` in Finish(), so a reader
// never sees a partially written file.
class PackedLeavesWriter {
 public:
  static StatusOr<std::unique_ptr<PackedLeavesWriter>> Create(
      const std::string& path, uint32_t num_blocks, size_t num_leaves,
      uint64_t tokenization_fingerprint);
  ~PackedLeavesWriter();

  PackedLeavesWriter(const PackedLeavesWriter&) = delete;
  PackedLeavesWriter& operator=(const PackedLeavesWriter&) = delete;

  // Leaves must be appended in order; `bytes` must hold at least
  // PackedLeafBytes(num_datapoints, num_blocks) bytes.
  Status AppendLeaf(uint32_t num_datapoints, ConstSpan<uint8_t> bytes);

  Status Finish();

 private:
  PackedLeavesWriter() = default;
  Status WriteAll(const void* data, size_t size);
  Status Flush();

  std::string path_;
  std::string tmp_path_;
  int fd_ = -1;
  uint32_t num_blocks_ = 0;
  size_t num_leaves_ = 0;
  uint64_t fingerprint_ = 0;
  uint64_t data_offset_ = 0;
  std::vector<uint32_t> leaf_sizes_;
  std::vector<char> buffer_;
};

}  // namespace research_scann

#endif
