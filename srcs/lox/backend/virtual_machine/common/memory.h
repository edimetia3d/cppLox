//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_MEMORY_H_
#define CLOX_SRCS_CLOX_MEMORY_H_

#include <cstdlib>
#include <cstring>

#include "lox/backend/virtual_machine/common/common.h"
namespace lox {
namespace vm {

void *reallocate(void *buffer, int old_size, int new_size);

template <class T, int SmallOpt = 128 / sizeof(T)>
struct CustomVec {
  CustomVec() { buffer_ = small_opt_buffer_; }
  void push_buffer(const T *bytes_buffer, int n);

  void push_back(const T &value) { push_buffer(&value, 1); }

  int size() const { return element_count_; }
  int byte_size() { return element_count_ * sizeof(T); }

  T &operator[](int index) { return buffer_[index]; }

  T *data();
  const T *data() const;
  ~CustomVec();
  bool operator==(const CustomVec &rhs) const;

  void reserve(int min_size) {
    if (min_size >= element_capacity_) {
      grow_capacity(min_size);
    }
  }

 private:
  void grow_capacity(int min_size);
  int element_count_ = 0;
  int element_capacity_ = SmallOpt;
  T *buffer_ = nullptr;
  T small_opt_buffer_[SmallOpt];
};

template <class T, int SmallOpt>
void CustomVec<T, SmallOpt>::grow_capacity(int min_size) {
  int old_capacity_ = element_capacity_;
  element_capacity_ = ((min_size + 7) / 8) * 16;
  if (buffer_ && buffer_ == small_opt_buffer_) {
    buffer_ = nullptr;
  }
  buffer_ = static_cast<T *>(reallocate(buffer_, old_capacity_ * sizeof(T), element_capacity_ * sizeof(T)));
}
template <class T, int SmallOpt>
CustomVec<T, SmallOpt>::~CustomVec() {
  if (buffer_ != small_opt_buffer_) {
    free(buffer_);
  }
}
template <class T, int SmallOpt>
T *CustomVec<T, SmallOpt>::data() {
  return buffer_;
}

template <class T, int SmallOpt>
const T *CustomVec<T, SmallOpt>::data() const {
  return buffer_;
}

template <class T, int SmallOpt>
void CustomVec<T, SmallOpt>::push_buffer(const T *bytes_buffer, int n) {
  if ((element_count_ + n) > element_capacity_) {
    grow_capacity(element_count_ + n);
  }
  memcpy(buffer_ + element_count_, bytes_buffer, n * sizeof(T));
  element_count_ += n;
}
template <class T, int SmallOpt>
bool CustomVec<T, SmallOpt>::operator==(const CustomVec &rhs) const {
  if (&rhs == this) {
    return true;
  }
  if (rhs.size() != size()) {
    return false;
  }
  for (int i = 0; i < size(); ++i) {
    if (rhs.buffer_[i] != buffer_[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_MEMORY_H_
