/*!
 * Copyright 2015-2019 by Contributors
 * \file common.h
 * \brief Threading utilities
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_H_
#define XGBOOST_COMMON_THREADING_UTILS_H_

#include <vector>
#include <algorithm>

namespace xgboost {
namespace common {

// Represent simple range of indexes [begin, end)
// Inspired by tbb::blocked_range
class Range1d {
 public:
  Range1d(size_t begin, size_t end): begin_(begin), end_(end) {
    CHECK_LT(begin, end);
  }

  size_t begin() {
    return begin_;
  }

  size_t end() {
    return end_;
  }

 private:
  size_t begin_;
  size_t end_;
};


// Split 2d space to balanced blocks
// Implementation of the class is inspired by tbb::blocked_range2d
// However, TBB provides only (n x m) 2d range (matrix) separated by blocks. Example:
// [ 1,2,3 ]
// [ 4,5,6 ]
// [ 7,8,9 ]
// But the class is able to work with different sizes in each 'row'. Example:
// [ 1,2 ]
// [ 3,4,5,6 ]
// [ 7,8,9]
// If grain_size is 2: It produces following blocks:
// [1,2], [3,4], [5,6], [7,8], [9]
// The class helps to process data in several tree nodes (non-balanced usually) in parallel
// Using nested parallelism (by nodes and by data in each node)
// it helps  to improve CPU resources utilization
class BlockedSpace2d {
 public:
  // Example of space:
  // [ 1,2 ]
  // [ 3,4,5,6 ]
  // [ 7,8,9]
  // BlockedSpace2d will create following blocks (tasks) if grain_size=2:
  // 1-block: first_dimension = 0, range of indexes in a 'row' = [0,2) (includes [1,2] values)
  // 2-block: first_dimension = 1, range of indexes in a 'row' = [0,2) (includes [3,4] values)
  // 3-block: first_dimension = 1, range of indexes in a 'row' = [2,4) (includes [5,6] values)
  // 4-block: first_dimension = 2, range of indexes in a 'row' = [0,2) (includes [7,8] values)
  // 5-block: first_dimension = 2, range of indexes in a 'row' = [2,3) (includes [9] values)
  // Arguments:
  // dim1 - size of the first dimension in the space
  // getter_size_dim2 - functor to get the second dimensions for each 'row' by row-index
  // grain_size - max size of produced blocks
  template<typename Func>
  BlockedSpace2d(size_t dim1, Func getter_size_dim2, size_t grain_size) {
    bounds_.resize(dim1 + 1);
    bounded_sized_dim_.resize(dim1);
    grain_size_ = grain_size;
    bounds_[0] = 0;

    for (size_t i = 0; i < dim1; ++i) {
      const size_t size = getter_size_dim2(i);
      const size_t n_blocks = size/grain_size + !!(size % grain_size);

      // first_dimension_.reserve(first_dimension_.size() + n_blocks);
      // ranges_.reserve(ranges_.size() + n_blocks);
      bounded_sized_dim_[i] = size;
      bounds_[i+1] = bounds_[i] + n_blocks;
    //   for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
    //     const size_t begin = iblock * grain_size;
    //     const size_t end   = std::min(begin + grain_size, size);
    //     AddBlock(i, begin, end);
    //   }
    }
  }

  // Amount of blocks(tasks) in a space
  size_t Size() const {
    return bounds_.back();
    // return ranges_.size();
  }

  // get index of the first dimension of i-th block(task)
  size_t GetFirstDimension(size_t i) const {
    auto first = std::upper_bound(bounds_.begin(), bounds_.end(), i) - 1;
    return first - bounds_.begin();
    // CHECK_LT(i, first_dimension_.size());
    // return first_dimension_[i];
  }

  // get a range of indexes for the second dimension of i-th block(task)
  Range1d GetRange(size_t i) const {
    auto first = std::upper_bound(bounds_.begin(), bounds_.end(), i) - 1;

    size_t idx = (first - bounds_.begin());

    size_t iblock = i - (*first);
    size_t begin = iblock * grain_size_;
    return Range1d(begin, std::min(begin + grain_size_, bounded_sized_dim_[idx]));

    // CHECK_LT(i, ranges_.size());
    // return ranges_[i];
  }

 private:
  void AddBlock(size_t first_dimension, size_t begin, size_t end) {
    first_dimension_.push_back(first_dimension);
    ranges_.emplace_back(begin, end);
  }

  std::vector<Range1d> ranges_;
  std::vector<size_t> first_dimension_;
  std::vector<size_t> bounds_;
  std::vector<size_t> bounded_first_dim_;
  std::vector<size_t> bounded_sized_dim_;
  size_t grain_size_;
};


// Wrapper to implement nested parallelism with simple omp parallel for
template<typename Func>
void ParallelFor2d(const BlockedSpace2d& space, Func func) {
  const int num_blocks_in_space = static_cast<int>(space.Size());

  #pragma omp parallel for
  for (auto i = 0; i < num_blocks_in_space; i++) {
    func(space.GetFirstDimension(i), space.GetRange(i));
  }
}

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADING_UTILS_H_
