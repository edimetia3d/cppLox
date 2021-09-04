//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_MEMORY_H_
#define CLOX_SRCS_CLOX_MEMORY_H_

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <cstring>

#include "lox/backend/virtual_machine/common/common.h"
namespace lox {
namespace vm {

template <class T>
concept TrivialCopyable = std::is_trivially_copyable_v<T>;

template <TrivialCopyable T, int SmallOpt = 128 / sizeof(T)>
struct Buffer {
  Buffer() { state_.buffer = state_.small_opt_buffer; }
  Buffer(const Buffer &) = delete;
  Buffer(Buffer &&rhs) noexcept { *this = std::move(rhs); }
  Buffer &operator=(Buffer &&rhs) noexcept {
    state_ = rhs.state_;
    rhs.state_.buffer = nullptr;
    return *this;
  }
  void push_buffer(const T *bytes_buffer, int n);

  void push_back(const T &value) { push_buffer(&value, 1); }

  int size() const { return state_.element_count; }
  void resize(int new_size);
  int byte_size() { return state_.element_count * sizeof(T); }

  T &operator[](int index) { return state_.buffer[index]; }

  T *data();
  const T *data() const;
  ~Buffer();
  bool operator==(const Buffer &rhs) const;

  void reserve(int min_size) {
    if (min_size >= state_.element_capacity) {
      modify_capacity(min_size, 1);
    }
  }

 private:
  void modify_capacity(int min_size, int reserve_ratio = 2);
  void SafeFree() {
    if (state_.buffer != state_.small_opt_buffer) {
      free(state_.buffer);
      state_.buffer = nullptr;
    }
  }
  struct CoreState {
    int element_count = 0;
    int element_capacity = SmallOpt;
    T *buffer = nullptr;
    T small_opt_buffer[SmallOpt];
  };
  CoreState state_;
};

template <TrivialCopyable T, int SmallOpt>
void Buffer<T, SmallOpt>::modify_capacity(int min_size, int reserve_ratio) {
  int new_capacity = ((min_size + 7) / 8) * 8 * reserve_ratio;
  T *new_buffer = nullptr;
  if (new_capacity < SmallOpt) {
    new_buffer = state_.small_opt_buffer;
    new_capacity = SmallOpt;
  } else {
    new_buffer = static_cast<T *>(malloc(new_capacity * sizeof(T)));
  }
  memcpy(new_buffer, state_.buffer, std::min(state_.element_count, state_.element_capacity) * sizeof(T));
  SafeFree();
  state_.buffer = new_buffer;
  state_.element_capacity = new_capacity;
}
template <TrivialCopyable T, int SmallOpt>
Buffer<T, SmallOpt>::~Buffer() {
  SafeFree();
}
template <TrivialCopyable T, int SmallOpt>
T *Buffer<T, SmallOpt>::data() {
  return state_.buffer;
}

template <TrivialCopyable T, int SmallOpt>
const T *Buffer<T, SmallOpt>::data() const {
  return state_.buffer;
}

template <TrivialCopyable T, int SmallOpt>
void Buffer<T, SmallOpt>::push_buffer(const T *bytes_buffer, int n) {
  if ((state_.element_count + n) > state_.element_capacity) {
    modify_capacity(state_.element_count + n, 2);
  }
  memcpy(state_.buffer + state_.element_count, bytes_buffer, n * sizeof(T));
  state_.element_count += n;
}
template <TrivialCopyable T, int SmallOpt>
bool Buffer<T, SmallOpt>::operator==(const Buffer &rhs) const {
  if (&rhs == this) {
    return true;
  }
  if (rhs.size() != size()) {
    return false;
  }
  for (int i = 0; i < size(); ++i) {
    if (rhs.state_.buffer[i] != state_.buffer[i]) {
      return false;
    }
  }
  return true;
}
template <TrivialCopyable T, int SmallOpt>
void Buffer<T, SmallOpt>::resize(int new_size) {
  if (new_size < state_.element_capacity) {
    if (new_size < state_.element_capacity / 10) {
      modify_capacity(new_size);
    }
  } else {
    modify_capacity(new_size);
  }
  state_.element_count = new_size;
}

}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_MEMORY_H_
