//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_HASH_MAP_H_
#include "lox/backend/virtual_machine/common/buffer.h"
namespace lox {
namespace vm {
template <class KeyT, class ValueT, uint32_t HashFn(KeyT)>
class HashMap {
 private:
  enum class EntryMark { FREE_TO_USE_MARK, TOMBSTONE_MARK, USED_MARK };

 public:
  struct Entry {
    KeyT key;
    ValueT value;
    EntryMark mark = EntryMark::FREE_TO_USE_MARK;
    bool IsFreeToUse() const { return mark == EntryMark::FREE_TO_USE_MARK || mark == EntryMark::TOMBSTONE_MARK; }
    bool IsTombStone() const { return mark == EntryMark::TOMBSTONE_MARK; }
    void MarkTomb() { mark = EntryMark::TOMBSTONE_MARK; }
  };

  HashMap(int capacity) : capacity(capacity) {
    entries.reserve(capacity);
    for (int i = 0; i < capacity; ++i) {
      entries[i].mark = EntryMark::FREE_TO_USE_MARK;
    }
  }

  bool Set(KeyT key, ValueT value) {
    if ((count + 1) > (capacity * TABLE_MAX_LOAD)) {
      AdJustCapacity(capacity * 2);
    }
    Entry* entry = FindInsertEntry(key);
    bool new_key_insert = entry->IsFreeToUse();
    if (new_key_insert && !entry->IsTombStone()) count++;

    entry->key = key;
    entry->value = value;
    entry->mark = EntryMark::USED_MARK;
    return new_key_insert;
  }
  Entry* Get(KeyT key) {
    if (count == 0) return nullptr;

    Entry* entry = FindInsertEntry(key);
    if (entry->IsFreeToUse()) return nullptr;

    return entry;
  }

  bool Del(KeyT key) {
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
    tmp.Merge(*this);
    *this = std::move(tmp);
  }
  Entry* FindInsertEntry(KeyT key) {
    uint32_t index = HashFn(key) % capacity;
    Entry* tombstone = nullptr;
    for (;;) {
      Entry* entry = &entries[index];
      if (entry->key == key) {
        return entry;
      } else {
        if ((entry->IsFreeToUse())) {
          if (entry->IsTombStone()) {
            tombstone = tombstone ? tombstone : entry;  // only use first tombstone
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
