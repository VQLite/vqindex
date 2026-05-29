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

#include "scann/utils/io_npy.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace research_scann {
namespace {

TEST(IoNpyTest, ParsesShapeDimensionsLargerThanInt32) {
  const std::string header =
      std::string("\x93NUMPY\x01\x00", 8) +
      "{'descr':'|u1', 'fortran_order':False, "
      "'shape':(4294967296,), } \n";

  size_t word_size = 0;
  std::vector<size_t> shape;
  bool fortran_order = true;
  Status status =
      internal::ParseNumpyHeader(header, &word_size, &shape, &fortran_order);

  EXPECT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(word_size, sizeof(uint8_t));
  EXPECT_FALSE(fortran_order);
  ASSERT_EQ(shape.size(), 1);
  EXPECT_EQ(shape[0], static_cast<size_t>(4294967296ULL));
}

TEST(IoNpyTest, RejectsOverflowingShapeDimensions) {
  const std::string header =
      std::string("\x93NUMPY\x01\x00", 8) +
      "{'descr':'|u1', 'fortran_order':False, "
      "'shape':(999999999999999999999999999999,), } \n";

  size_t word_size = 0;
  std::vector<size_t> shape;
  bool fortran_order = false;
  Status status =
      internal::ParseNumpyHeader(header, &word_size, &shape, &fortran_order);

  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace research_scann
