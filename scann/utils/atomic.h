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

#ifndef SCANN_UTILS_ATOMIC_H_
#define SCANN_UTILS_ATOMIC_H_

#include <atomic>

namespace research_scann {

template <typename T>
class RelaxedAtomic {
 public:
  RelaxedAtomic() = default;
  explicit RelaxedAtomic(T value) : value_(value) {}
  RelaxedAtomic(const RelaxedAtomic& other)
      : value_(other.value_.load(std::memory_order_relaxed)) {}
  RelaxedAtomic& operator=(const RelaxedAtomic& other) {
    value_.store(other.value_.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    return *this;
  }
  RelaxedAtomic(RelaxedAtomic&& other) noexcept
      : value_(other.value_.load(std::memory_order_relaxed)) {}
  RelaxedAtomic& operator=(RelaxedAtomic&& other) noexcept {
    value_.store(other.value_.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    return *this;
  }
  explicit operator T() const { return value_.load(std::memory_order_relaxed); }
  T load() const { return value_.load(std::memory_order_relaxed); }
  void store(T value) { value_.store(value, std::memory_order_relaxed); }

 private:
  std::atomic<T> value_;
};

}  // namespace research_scann

#endif
