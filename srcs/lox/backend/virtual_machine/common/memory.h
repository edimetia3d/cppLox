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

  int size() { return element_count_; }
  int byte_size() { return element_count_ * sizeof(T); }

  T &operator[](int index) { return buffer_[index]; }

  T *data();

  ~CustomVec();

 private:
  void grow_capacity();
  int element_count_ = 0;
  int element_capacity_ = SmallOpt;
  T *buffer_ = nullptr;
  T small_opt_buffer_[SmallOpt];
};

template <class T, int SmallOpt>
void CustomVec<T, SmallOpt>::grow_capacity() {
  int old_capacity_ = element_capacity_;
  constexpr int MINSIZE = (SmallOpt == 0 ? 8 : SmallOpt);
  element_capacity_ = (element_capacity_ < MINSIZE ? MINSIZE : element_capacity_ * 2);
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
void CustomVec<T, SmallOpt>::push_buffer(const T *bytes_buffer, int n) {
  if ((element_count_ + n) > element_capacity_) {
    grow_capacity();
  }
  memcpy(buffer_ + element_count_, bytes_buffer, n * sizeof(T));
  element_count_ += n;
}
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_MEMORY_H_
