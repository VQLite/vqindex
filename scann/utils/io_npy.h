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

#ifndef SCANN_UTILS_IO_NPY_H_
#define SCANN_UTILS_IO_NPY_H_

#include <cctype>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
std::string numpy_type_name();

namespace internal {

inline bool ParseSizeT(absl::string_view value, size_t* result) {
  if (value.empty()) return false;

  size_t parsed = 0;
  for (char c : value) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    const size_t digit = c - '0';
    if (parsed >
        (std::numeric_limits<size_t>::max() - digit) / static_cast<size_t>(10)) {
      return false;
    }
    parsed = parsed * 10 + digit;
  }

  *result = parsed;
  return true;
}

inline Status ParseNumpyHeader(absl::string_view header, size_t* word_size,
                               vector<size_t>* shape, bool* fortran_order) {
  constexpr absl::string_view kMagic = "\x93NUMPY";
  if (header.find(kMagic) != 0) {
    return InvalidArgumentError("Invalid numpy header magic.");
  }

  const size_t descr_pos = header.find("descr");
  if (descr_pos == absl::string_view::npos) {
    return InvalidArgumentError(
        "Numpy header is missing required field: descr.");
  }
  const size_t descr_colon = header.find(':', descr_pos);
  const size_t descr_quote = header.find_first_of("'\"", descr_colon);
  if (descr_colon == absl::string_view::npos ||
      descr_quote == absl::string_view::npos) {
    return InvalidArgumentError("Invalid numpy descr field.");
  }
  const size_t descr_end = header.find(header[descr_quote], descr_quote + 1);
  if (descr_end == absl::string_view::npos) {
    return InvalidArgumentError("Invalid numpy descr field.");
  }

  absl::string_view descr =
      header.substr(descr_quote + 1, descr_end - descr_quote - 1);
  size_t word_start = descr.size();
  while (word_start > 0 &&
         std::isdigit(static_cast<unsigned char>(descr[word_start - 1]))) {
    --word_start;
  }
  if (word_start == descr.size() ||
      !ParseSizeT(descr.substr(word_start), word_size)) {
    return InvalidArgumentError("Invalid numpy descr word size.");
  }

  const size_t fortran_pos = header.find("fortran_order");
  if (fortran_pos == absl::string_view::npos) {
    return InvalidArgumentError(
        "Numpy header is missing required field: fortran_order.");
  }
  const size_t fortran_colon = header.find(':', fortran_pos);
  if (fortran_colon == absl::string_view::npos) {
    return InvalidArgumentError("Invalid numpy fortran_order field.");
  }
  size_t fortran_value = fortran_colon + 1;
  while (fortran_value < header.size() &&
         std::isspace(static_cast<unsigned char>(header[fortran_value]))) {
    ++fortran_value;
  }
  if (header.substr(fortran_value, 4) == "True") {
    *fortran_order = true;
  } else if (header.substr(fortran_value, 5) == "False") {
    *fortran_order = false;
  } else {
    return InvalidArgumentError("Invalid numpy fortran_order value.");
  }

  const size_t shape_pos = header.find("shape");
  if (shape_pos == absl::string_view::npos) {
    return InvalidArgumentError(
        "Numpy header is missing required field: shape.");
  }
  const size_t shape_start = header.find('(', shape_pos);
  const size_t shape_end = header.find(')', shape_start);
  if (shape_start == absl::string_view::npos ||
      shape_end == absl::string_view::npos || shape_end < shape_start) {
    return InvalidArgumentError("Invalid numpy shape field.");
  }

  shape->clear();
  absl::string_view shape_text =
      header.substr(shape_start + 1, shape_end - shape_start - 1);
  size_t pos = 0;
  while (pos < shape_text.size()) {
    while (pos < shape_text.size() &&
           (std::isspace(static_cast<unsigned char>(shape_text[pos])) ||
            shape_text[pos] == ',')) {
      ++pos;
    }
    if (pos >= shape_text.size()) break;

    const size_t number_start = pos;
    while (pos < shape_text.size() &&
           std::isdigit(static_cast<unsigned char>(shape_text[pos]))) {
      ++pos;
    }
    if (number_start == pos) {
      return InvalidArgumentError("Invalid numpy shape dimension.");
    }

    size_t dim = 0;
    if (!ParseSizeT(shape_text.substr(number_start, pos - number_start),
                    &dim)) {
      return InvalidArgumentError("Numpy shape dimension is too large.");
    }
    shape->push_back(dim);
  }

  return OkStatus();
}

}  // namespace internal

template <typename T>
Status SpanToNumpy(absl::string_view filename, ConstSpan<T> data,
                   ConstSpan<size_t> dim_sizes = {},
                   DimensionIndex last_dim = kInvalidDimension) {
  std::string shape_str = "(";
  size_t dim_prod = 1;
  for (size_t dim_size : dim_sizes) {
    dim_prod *= dim_size;
    shape_str += std::to_string(dim_size) + ",";
  }
  if (last_dim != kInvalidDimension) {
    shape_str += std::to_string(last_dim) + ",)";
  } else {
    if (dim_prod == 0 || data.size() % dim_prod != 0)
      return InvalidArgumentError(
          "Size of data isn't compatible with given shape");
    shape_str += std::to_string(data.size() / dim_prod) + ",)";
  }

  if (shape_str.size() > 65000)
    return InvalidArgumentError("Shape string is too large for npy format: " +
                                shape_str);

  std::string pt1("\x93NUMPY\x01\x00  ", 10);
  std::string dict =
      absl::StrFormat("{'descr':%s, 'fortran_order':False, 'shape':%s}",
                      numpy_type_name<T>(), shape_str);
  while ((pt1.size() + dict.size()) % 64 != 63) dict += " ";
  dict += "\n";
  pt1[8] = dict.size() % 256;
  pt1[9] = dict.size() / 256;
  const std::string header = pt1 + dict;

  OpenSourceableFileWriter writer(filename);
  SCANN_RETURN_IF_ERROR(
      writer.Write(ConstSpan<char>(header.data(), header.size())));

  const char* ptr = reinterpret_cast<const char*>(data.data());
  return writer.Write(ConstSpan<char>(ptr, data.size() * sizeof(T)));
}

template <typename T>
Status VectorToNumpy(absl::string_view filename, const vector<T>& data,
                     const vector<size_t>& dim_sizes = {}) {
  return SpanToNumpy(filename, ConstSpan<T>(data.data(), data.size()),
                     ConstSpan<size_t>(dim_sizes.data(), dim_sizes.size()));
}

template <typename T>
Status DatasetToNumpy(absl::string_view filename, const DenseDataset<T>& data) {
  return SpanToNumpy(filename, data.data(), {data.size()},
                     data.dimensionality());
}

template <typename T>
StatusOr<pair<std::vector<T>, std::vector<size_t>>> NumpyToVectorAndShape(
    absl::string_view filename) {
  OpenSourceableFileReader reader(filename);
  std::string header;
  SCANN_RETURN_IF_ERROR(reader.ReadLine(header));

  size_t word_size;
  vector<size_t> shape;
  bool fortran_order;
  SCANN_RETURN_IF_ERROR(
      internal::ParseNumpyHeader(header, &word_size, &shape, &fortran_order));
  if (fortran_order) return FailedPreconditionError("Numpy file isn't C-style");
  if (word_size != sizeof(T))
    return FailedPreconditionError("word_size != sizeof(T): %d != %d",
                                   word_size, sizeof(T));

  size_t total_size = 1;
  for (size_t s : shape) {
    if (s != 0 && total_size > std::numeric_limits<size_t>::max() / s) {
      return InvalidArgumentError("Numpy shape total size is too large.");
    }
    total_size *= s;
  }
  vector<T> buffer(total_size);
  SCANN_RETURN_IF_ERROR(reader.Read(total_size * sizeof(T),
                                    reinterpret_cast<char*>(buffer.data())));
  return std::make_pair(std::move(buffer), std::move(shape));
}

}  // namespace research_scann

#endif
