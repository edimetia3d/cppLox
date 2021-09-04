//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#include "lox/backend/virtual_machine/common/buffer.h"
namespace lox {
namespace vm {
template <class KeyT, class ValueT, uint32_t HashFn(KeyT), KeyT FREE_TO_USE_MARK, ValueT TOMBSTONE_MARK>
class HashMap {
 public:
  struct Entry {
    KeyT key = FREE_TO_USE_MARK;
    ValueT value;
    bool IsFreeToUse() { return key == FREE_TO_USE_MARK; }
    bool IsTombStone() { return key == FREE_TO_USE_MARK && value == TOMBSTONE_MARK; }
    bool MarkTomb() {
      key = FREE_TO_USE_MARK;
      value = TOMBSTONE_MARK;
    }
  };

  HashMap(int capacity) : capacity(capacity) {
    entries.reserve(capacity);
    for (int i = 0; i < capacity; ++i) {
      entries[i].key = FREE_TO_USE_MARK;
      assert(entries[i].value != TOMBSTONE_MARK);
    }
  }

  bool Set(KeyT key, ValueT value) {
    assert(key != FREE_TO_USE_MARK);
    assert(value != TOMBSTONE_MARK);
    if ((count + 1) > (capacity * TABLE_MAX_LOAD)) {
      AdJustCapacity(capacity * 2);
    }
    Entry* entry = FindInsertEntry();
    bool new_key_insert = entry->IsFreeToUse();
    if (new_key_insert && !entry->IsTombStone()) count++;

    entry->key = key;
    entry->value = value;
    return new_key_insert;
  }
  bool Get(KeyT key, ValueT* value) {
    assert(key != FREE_TO_USE_MARK);
    if (count == 0) return false;

    Entry* entry = FindInsertEntry(key);
    if (entry->IsFreeToUse()) return false;

    *value = entry->value;
    return true;
  }

  bool Del(KeyT key) {
    assert(key != FREE_TO_USE_MARK);
    Entry* entry = FindInsertEntry(key);
    if (!entry->IsFreeToUse()) {
      entry->MarkTomb();
      return true;
    }
    printf("Warning: key not found when delete.\n");
    return false;
  }

  void Merge(const HashMap& map) {
    for (int i = 0; i < map.capacity; ++i) {
      if (!map.entries[i].IsFreeToUse()) {
        Set(map.entries[i].key, map.entries[i].value);
      }
    }
  }

 private:
  void AdJustCapacity(int new_capacity) {
    HashMap tmp(new_capacity);
    for (int i = 0; i < capacity; ++i) {
      if (!entries[i].IsFreeToUse()) {
        tmp.Set(entries[i].key, entries[i].value);
      }
    }
    *this = std::move(tmp);
  }
  Entry FindInsertEntry(KeyT key) {
    uint32_t index = HashFn(key) % capacity;
    Entry* tombstone = nullptr;
    for (;;) {
      Entry* entry = &entries[index];
      if (entry->key == key) {
        return entry;
      } else {
        if ((entry->IsFreeToUse())) {
          if (entry->IsTombStone()) {
            tombstone = entry;
          } else {
            return tombstone ? tombstone : entry;
          }
        }
      }
      index = (index + 1) % capacity;
    }
  }

 private:
  static constexpr double TABLE_MAX_LOAD = 0.7;
  int count = 0;
  int capacity = 0;
  Buffer<Entry, 0> entries;
};
}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
