//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#include "lox/backend/virtual_machine/common/vector.h"
namespace lox {
namespace vm {
template <class KeyT, class ValueT>
class HashMap {
 public:
  HashMap(int capacity) { Reserve(capacity); }
  void Reserve(int capacity) {
    capacity = capacity;
    entries.reserve(capacity);
  }
  struct Entry {
    KeyT key;
    ValueT value;
  };

 private:
  int count = 0;
  int capacity = 0;
  Vector<Entry, 0> entries;
};
}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
