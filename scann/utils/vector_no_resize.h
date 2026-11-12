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

#ifndef SCANN_UTILS_VECTOR_NO_RESIZE_H_
#define SCANN_UTILS_VECTOR_NO_RESIZE_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace research_scann {

template <typename T, typename A = std::allocator<T>>
class DefaultInitAllocator : public A {
  using a_t = std::allocator_traits<A>;

 public:
  template <typename U>
  struct rebind {
    using other =
        DefaultInitAllocator<U, typename a_t::template rebind_alloc<U>>;
  };

  using A::A;
  DefaultInitAllocator() = default;
  template <typename U>
  explicit DefaultInitAllocator(const DefaultInitAllocator<U>& other)
      : A(other) {}

  template <typename U>
  void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void*>(ptr)) U;
  }

  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
  }
};

template <typename T>
class VectorNoResize {
 public:
  using allocator_type = DefaultInitAllocator<T>;
  using vector_type = std::vector<T, allocator_type>;
  using value_type = typename vector_type::value_type;
  using size_type = typename vector_type::size_type;
  using difference_type = typename vector_type::difference_type;
  using reference = typename vector_type::reference;
  using const_reference = typename vector_type::const_reference;
  using pointer = typename vector_type::pointer;
  using const_pointer = typename vector_type::const_pointer;
  using iterator = typename vector_type::iterator;
  using const_iterator = typename vector_type::const_iterator;

  VectorNoResize() = default;
  explicit VectorNoResize(const allocator_type& alloc) : vec_(alloc) {}

  reference operator[](size_type pos) { return vec_[pos]; }
  const_reference operator[](size_type pos) const { return vec_[pos]; }

  reference at(size_type pos) { return vec_.at(pos); }
  const_reference at(size_type pos) const { return vec_.at(pos); }

  reference front() { return vec_.front(); }
  const_reference front() const { return vec_.front(); }

  reference back() { return vec_.back(); }
  const_reference back() const { return vec_.back(); }

  pointer data() noexcept { return vec_.data(); }
  const_pointer data() const noexcept { return vec_.data(); }

  iterator begin() noexcept { return vec_.begin(); }
  const_iterator begin() const noexcept { return vec_.begin(); }
  const_iterator cbegin() const noexcept { return vec_.cbegin(); }

  iterator end() noexcept { return vec_.end(); }
  const_iterator end() const noexcept { return vec_.end(); }
  const_iterator cend() const noexcept { return vec_.cend(); }

  bool empty() const noexcept { return vec_.empty(); }
  size_type size() const noexcept { return vec_.size(); }
  size_type max_size() const noexcept { return vec_.max_size(); }

  void reserve(size_type new_cap) { vec_.reserve(new_cap); }
  size_type capacity() const noexcept { return vec_.capacity(); }
  void shrink_to_fit() { vec_.shrink_to_fit(); }

  void clear() noexcept { vec_.clear(); }

  void push_back(const T& value) { vec_.push_back(value); }
  void push_back(T&& value) { vec_.push_back(std::move(value)); }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    return vec_.emplace_back(std::forward<Args>(args)...);
  }

  void pop_back() { vec_.pop_back(); }

  void swap(VectorNoResize& other) noexcept { vec_.swap(other.vec_); }

  void resize_uninitialized(size_type new_size) { vec_.resize(new_size); }

  void resize_and_initialize(size_type new_size, const T& value) {
    vec_.resize(new_size, value);
  }

  operator absl::Span<T>() { return absl::MakeSpan(vec_); }
  operator absl::Span<const T>() const { return absl::MakeConstSpan(vec_); }

 private:
  vector_type vec_;
};

template <typename T>
void swap(VectorNoResize<T>& lhs, VectorNoResize<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace research_scann

#endif
