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



#ifndef SCANN_UTILS_HEAP_TOPN_H_
#define SCANN_UTILS_HEAP_TOPN_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/log/check.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T, typename Cmp = std::greater<T>,
          template <typename U> class VectorType = std::vector>
class HeapTopN {
 public:
  enum class State : uint8_t { unordered, bottom_known, heap_sorted };

  using value_type = T;

  using UnsortedIterator = typename VectorType<value_type>::const_iterator;

  explicit HeapTopN(DatapointIndex limit) : HeapTopN(limit, Cmp()) {}
  HeapTopN(DatapointIndex limit, const Cmp &cmp) : limit_(limit), cmp_(cmp) {}

  HeapTopN(const HeapTopN &top_n) = default;
  HeapTopN &operator=(const HeapTopN &top_n) = default;

  HeapTopN(HeapTopN &&other) noexcept
      : elements_(std::move(other.elements_)),
        limit_(other.limit_),
        state_(other.state_),
        cmp_(other.cmp_) {
    other.Reset();
  }

  HeapTopN &operator=(HeapTopN &&other) noexcept {
    elements_ = std::move(other.elements_);
    limit_ = other.limit_;
    cmp_ = other.cmp_;
    state_ = other.state_;
    other.Reset();
    return *this;
  }

  DatapointIndex limit() const { return limit_; }
  DatapointIndex size() const {
    return std::min<DatapointIndex>(elements_.size(), limit_);
  }
  bool empty() const { return size() == 0; }
  void reserve(DatapointIndex n) { elements_.reserve(std::min(n, limit_ + 1)); }

  void push(const value_type &v) { push(v, nullptr); }
  void push(const value_type &v, value_type *dropped) {
    PushInternal(v, dropped);
  }

  void push(value_type &&v) { push(std::move(v), nullptr); }
  void push(value_type &&v, value_type *dropped) {
    PushInternal(std::move(v), dropped);
  }

  const value_type &peek_bottom();

  VectorType<value_type> Take();
  VectorType<value_type> TakeUnsorted();
  VectorType<value_type> TakeNondestructive() const;
  VectorType<value_type> TakeUnsortedNondestructive() const;

  void ExtractNondestructive(VectorType<value_type> *output) const;
  void ExtractUnsortedNondestructive(VectorType<value_type> *output) const;
  UnsortedIterator unsorted_begin() const { return elements_.begin(); }
  UnsortedIterator unsorted_end() const { return elements_.begin() + size(); }

  const Cmp &key_comp() const { return cmp_; }

  void Reset() {
    elements_.clear();
    state_ = State::unordered;
  }

  void Reset(size_t limit) {
    Reset();
    limit_ = limit;
  }

 private:
  template <typename U>
  void PushInternal(U &&v, value_type *dropped);

  VectorType<value_type> elements_;
  DatapointIndex limit_;
  State state_ = State::unordered;
  Cmp cmp_;
};

template <typename T, typename Cmp, template <typename U> class VectorType>
template <typename U>
void HeapTopN<T, Cmp, VectorType>::PushInternal(U &&v, value_type *dropped) {
  if (limit_ == 0) {
    if (dropped) *dropped = std::forward<U>(v);
    return;
  }
  if (state_ != State::heap_sorted) {
    elements_.push_back(std::forward<U>(v));
    if (state_ == State::unordered ||
        cmp_(elements_.back(), elements_.front())) {
    } else {
      using std::swap;
      swap(elements_.front(), elements_.back());
    }
    if (elements_.size() == limit_ + 1) {
      std::make_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
      state_ = State::heap_sorted;
    }
  } else {
    if (cmp_(v, elements_.front())) {
      elements_.back() = std::forward<U>(v);
      std::push_heap(elements_.begin(), elements_.end(), cmp_);
      if (dropped) *dropped = std::move(elements_.front());
      std::pop_heap(elements_.begin(), elements_.end(), cmp_);
    } else {
      if (dropped) *dropped = std::forward<U>(v);
    }
  }
}

template <typename T, typename Cmp, template <typename U> class VectorType>
auto HeapTopN<T, Cmp, VectorType>::peek_bottom() -> const value_type & {
  CHECK(!empty());
  if (state_ == State::unordered) {
    auto min_iter = std::max_element(elements_.begin(), elements_.end(), cmp_);
    if (min_iter != elements_.begin()) {
      std::iter_swap(min_iter, elements_.begin());
    }
    state_ = State::bottom_known;
  }
  return elements_.front();
}

template <typename T, typename Cmp, template <typename U> class VectorType>
auto HeapTopN<T, Cmp, VectorType>::Take() -> VectorType<value_type> {
  VectorType<value_type> out = std::move(elements_);
  if (state_ != State::heap_sorted) {
    std::sort(out.begin(), out.end(), cmp_);
  } else {
    out.pop_back();
    std::sort_heap(out.begin(), out.end(), cmp_);
  }
  Reset();
  return out;
}

template <typename T, typename Cmp, template <typename U> class VectorType>
auto HeapTopN<T, Cmp, VectorType>::TakeUnsorted() -> VectorType<value_type> {
  VectorType<value_type> out = std::move(elements_);
  if (state_ == State::heap_sorted) {
    out.pop_back();
  }
  Reset();
  return out;
}

template <typename T, typename Cmp, template <typename U> class VectorType>
auto HeapTopN<T, Cmp, VectorType>::TakeNondestructive() const
    -> VectorType<value_type> {
  VectorType<value_type> out;
  ExtractNondestructive(&out);
  return out;
}

template <typename T, typename Cmp, template <typename U> class VectorType>
auto HeapTopN<T, Cmp, VectorType>::TakeUnsortedNondestructive() const
    -> VectorType<value_type> {
  VectorType<value_type> out;
  ExtractUnsortedNondestructive(&out);
  return out;
}

template <typename T, typename Cmp, template <typename U> class VectorType>
void HeapTopN<T, Cmp, VectorType>::ExtractNondestructive(
    VectorType<value_type> *output) const {
  CHECK(output);
  *output = elements_;
  if (state_ != State::heap_sorted) {
    std::sort(output->begin(), output->end(), cmp_);
  } else {
    output->pop_back();
    std::sort_heap(output->begin(), output->end(), cmp_);
  }
}

template <typename T, typename Cmp, template <typename U> class VectorType>
void HeapTopN<T, Cmp, VectorType>::ExtractUnsortedNondestructive(
    VectorType<value_type> *output) const {
  CHECK(output);
  *output = elements_;
  if (state_ == State::heap_sorted) {
    output->pop_back();
  }
}

}  // namespace research_scann

#endif
