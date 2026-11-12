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

#include "scann/utils/packed_leaves_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include "absl/strings/str_cat.h"

namespace research_scann {
namespace {

constexpr char kMagic[8] = {'V', 'Q', 'A', 'H', 'P', 'K', '1', '6'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 64;
constexpr size_t kWriteBufferBytes = 4 << 20;

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t num_blocks;
  uint32_t num_leaves;
  uint32_t reserved0;
  uint64_t num_datapoints;
  uint64_t fingerprint;
  uint64_t data_offset;
  uint64_t reserved1[2];
};
static_assert(sizeof(Header) == kHeaderBytes, "unexpected header layout");

uint64_t DataOffsetFor(size_t num_leaves) {
  const uint64_t end = kHeaderBytes + num_leaves * sizeof(uint32_t);
  return (end + 63) & ~uint64_t{63};
}

inline uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t MixPair(uint64_t token, uint64_t dp_idx) {
  return Mix((token << 32) ^ dp_idx);
}

uint64_t FinishFingerprint(uint64_t sum, uint64_t num_datapoints,
                           uint64_t num_leaves) {
  return Mix(sum ^ Mix(num_datapoints) ^ Mix(num_leaves << 40));
}

Status ErrnoError(const std::string& what, const std::string& path) {
  return InternalError(
      absl::StrCat(what, " ", path, ": ", std::strerror(errno)));
}

Status PReadAll(int fd, void* data, size_t size, uint64_t offset,
                const std::string& path) {
  char* p = static_cast<char*>(data);
  while (size > 0) {
    const ssize_t n = pread(fd, p, size, offset);
    if (n < 0) {
      if (errno == EINTR) continue;
      return ErrnoError("read", path);
    }
    if (n == 0) return absl::DataLossError(absl::StrCat(path, " is truncated."));
    p += n;
    size -= n;
    offset += n;
  }
  return OkStatus();
}

}  // namespace

// The fingerprint is a sum over (partition, datapoint) pairs, so it comes out
// the same from either representation of the tokenization.
uint64_t TokenizationFingerprint(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token) {
  uint64_t sum = 0, n = 0;
  for (size_t token = 0; token < datapoints_by_token.size(); ++token) {
    for (DatapointIndex dp_idx : datapoints_by_token[token]) {
      sum += MixPair(token, dp_idx);
    }
    n += datapoints_by_token[token].size();
  }
  return FinishFingerprint(sum, n, datapoints_by_token.size());
}

uint64_t TokenizationFingerprint(ConstSpan<int32_t> datapoint_to_token,
                                 size_t num_leaves) {
  uint64_t sum = 0;
  for (size_t dp_idx = 0; dp_idx < datapoint_to_token.size(); ++dp_idx) {
    sum += MixPair(static_cast<uint32_t>(datapoint_to_token[dp_idx]), dp_idx);
  }
  return FinishFingerprint(sum, datapoint_to_token.size(), num_leaves);
}

StatusOr<std::unique_ptr<PackedLeavesReader>> PackedLeavesReader::Open(
    const std::string& path) {
  std::unique_ptr<PackedLeavesReader> reader(new PackedLeavesReader());
  reader->path_ = path;
  reader->fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (reader->fd_ < 0) return ErrnoError("open", path);

  struct stat st;
  if (fstat(reader->fd_, &st) != 0) return ErrnoError("stat", path);
  const uint64_t file_size = st.st_size;
  if (file_size < kHeaderBytes) {
    return absl::DataLossError(absl::StrCat(path, " is truncated."));
  }

  Header header;
  SCANN_RETURN_IF_ERROR(
      PReadAll(reader->fd_, &header, sizeof(header), 0, path));
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
    return InvalidArgumentError(
        absl::StrCat(path, " is not a packed AH leaves file."));
  }
  if (header.version != kVersion) {
    return InvalidArgumentError(absl::StrCat(
        path, " has unsupported version ", header.version, "."));
  }
  if (header.data_offset != DataOffsetFor(header.num_leaves) ||
      header.data_offset > file_size) {
    return absl::DataLossError(absl::StrCat(path, " has an inconsistent header."));
  }

  reader->num_blocks_ = header.num_blocks;
  reader->num_datapoints_ = header.num_datapoints;
  reader->fingerprint_ = header.fingerprint;
  reader->leaf_sizes_.resize(header.num_leaves);
  SCANN_RETURN_IF_ERROR(PReadAll(reader->fd_, reader->leaf_sizes_.data(),
                                 header.num_leaves * sizeof(uint32_t),
                                 kHeaderBytes, path));

  reader->leaf_offsets_.resize(header.num_leaves);
  uint64_t offset = header.data_offset;
  uint64_t total = 0;
  for (size_t i = 0; i < header.num_leaves; ++i) {
    reader->leaf_offsets_[i] = offset;
    offset += PackedLeafBytes(reader->leaf_sizes_[i], header.num_blocks);
    total += reader->leaf_sizes_[i];
  }
  if (total != header.num_datapoints) {
    return absl::DataLossError(absl::StrCat(
        path, ": partition sizes add up to ", total, " datapoints, header says ",
        header.num_datapoints, "."));
  }
  if (offset != file_size) {
    return absl::DataLossError(absl::StrCat(path, " is ", file_size,
                                      " bytes, expected ", offset, "."));
  }
  return std::move(reader);
}

PackedLeavesReader::~PackedLeavesReader() {
  if (fd_ >= 0) close(fd_);
}

Status PackedLeavesReader::ReadLeaf(size_t leaf,
                                    std::vector<uint8_t>* out) const {
  if (leaf >= leaf_sizes_.size()) {
    return OutOfRangeError(absl::StrCat("Leaf ", leaf, " out of range."));
  }
  out->resize(PackedLeafBytes(leaf_sizes_[leaf], num_blocks_));
  if (out->empty()) return OkStatus();
  return PReadAll(fd_, out->data(), out->size(), leaf_offsets_[leaf], path_);
}

StatusOr<std::unique_ptr<PackedLeavesWriter>> PackedLeavesWriter::Create(
    const std::string& path, uint32_t num_blocks, size_t num_leaves,
    uint64_t tokenization_fingerprint) {
  std::unique_ptr<PackedLeavesWriter> writer(new PackedLeavesWriter());
  writer->path_ = path;
  writer->tmp_path_ = path + ".tmp";
  writer->num_blocks_ = num_blocks;
  writer->num_leaves_ = num_leaves;
  writer->fingerprint_ = tokenization_fingerprint;
  writer->data_offset_ = DataOffsetFor(num_leaves);
  writer->leaf_sizes_.reserve(num_leaves);
  writer->buffer_.reserve(kWriteBufferBytes);
  writer->fd_ = open(writer->tmp_path_.c_str(),
                     O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (writer->fd_ < 0) return ErrnoError("create", writer->tmp_path_);
  // The header and the size table are written in Finish(); leave their room.
  if (lseek(writer->fd_, writer->data_offset_, SEEK_SET) < 0) {
    return ErrnoError("seek", writer->tmp_path_);
  }
  return std::move(writer);
}

PackedLeavesWriter::~PackedLeavesWriter() {
  if (fd_ >= 0) {
    close(fd_);
    unlink(tmp_path_.c_str());
  }
}

Status PackedLeavesWriter::Flush() {
  const char* p = buffer_.data();
  size_t size = buffer_.size();
  while (size > 0) {
    const ssize_t n = write(fd_, p, size);
    if (n < 0) {
      if (errno == EINTR) continue;
      return ErrnoError("write", tmp_path_);
    }
    p += n;
    size -= n;
  }
  buffer_.clear();
  return OkStatus();
}

Status PackedLeavesWriter::WriteAll(const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  if (buffer_.size() + size > kWriteBufferBytes) {
    SCANN_RETURN_IF_ERROR(Flush());
  }
  if (size >= kWriteBufferBytes) {
    while (size > 0) {
      const ssize_t n = write(fd_, p, size);
      if (n < 0) {
        if (errno == EINTR) continue;
        return ErrnoError("write", tmp_path_);
      }
      p += n;
      size -= n;
    }
    return OkStatus();
  }
  buffer_.insert(buffer_.end(), p, p + size);
  return OkStatus();
}

Status PackedLeavesWriter::AppendLeaf(uint32_t num_datapoints,
                                      ConstSpan<uint8_t> bytes) {
  if (fd_ < 0) return FailedPreconditionError("Writer is closed.");
  if (leaf_sizes_.size() >= num_leaves_) {
    return FailedPreconditionError("More leaves appended than declared.");
  }
  const size_t expected = PackedLeafBytes(num_datapoints, num_blocks_);
  if (bytes.size() < expected) {
    return InvalidArgumentError(
        absl::StrCat("Leaf ", leaf_sizes_.size(), " has ", bytes.size(),
                     " packed bytes, expected ", expected, "."));
  }
  leaf_sizes_.push_back(num_datapoints);
  return WriteAll(bytes.data(), expected);
}

Status PackedLeavesWriter::Finish() {
  if (fd_ < 0) return FailedPreconditionError("Writer is closed.");
  if (leaf_sizes_.size() != num_leaves_) {
    return FailedPreconditionError(
        absl::StrCat(leaf_sizes_.size(), " leaves appended, ", num_leaves_,
                     " declared."));
  }
  SCANN_RETURN_IF_ERROR(Flush());

  Header header;
  std::memset(&header, 0, sizeof(header));
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.version = kVersion;
  header.num_blocks = num_blocks_;
  header.num_leaves = num_leaves_;
  header.fingerprint = fingerprint_;
  header.data_offset = data_offset_;
  for (uint32_t n : leaf_sizes_) header.num_datapoints += n;

  std::vector<char> front(data_offset_, 0);
  std::memcpy(front.data(), &header, sizeof(header));
  std::memcpy(front.data() + kHeaderBytes, leaf_sizes_.data(),
              leaf_sizes_.size() * sizeof(uint32_t));
  size_t done = 0;
  while (done < front.size()) {
    const ssize_t n =
        pwrite(fd_, front.data() + done, front.size() - done, done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return ErrnoError("write", tmp_path_);
    }
    done += n;
  }

  const int close_ret = close(fd_);
  fd_ = -1;
  if (close_ret != 0) {
    unlink(tmp_path_.c_str());
    return ErrnoError("close", tmp_path_);
  }
  if (std::rename(tmp_path_.c_str(), path_.c_str()) != 0) {
    unlink(tmp_path_.c_str());
    return ErrnoError("rename", tmp_path_);
  }
  return OkStatus();
}

}  // namespace research_scann
