//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#include <cassert>
#include <cstring>

#include "lox/backend/virtual_machine/common/buffer.h"
#include "lox/backend/virtual_machine/common/hash_map.h"
#include "lox/backend/virtual_machine/common/link_list.h"
namespace lox {
namespace vm {

enum class ObjType { UNKNOWN, OBJ_STRING, OBJ_FUNCTION };

struct Obj {
  ObjType type;

  template <class T>
  const T* const As() const {
    assert(type == T::TYPE_ID);
    return reinterpret_cast<const T*>(this);
  }
  template <class T>
  T* As() {
    assert(type == T::TYPE_ID);
    return reinterpret_cast<T*>(this);
  }
  template <class T>
  bool IsType() const {
    return type == T::TYPE_ID;
  }
  bool Equal(const Obj* rhs) const;
  void Print() const;
  static void Destroy(Obj* obj);

  static LinkList<Obj*>& AllCreatedObj();

 protected:
  Obj() : type(ObjType::UNKNOWN) { AllCreatedObj().Insert(this); };
};

class Chunk;
struct ObjFunction : public Obj {
  ObjFunction();
  constexpr static ObjType TYPE_ID = ObjType::OBJ_FUNCTION;
  int arity = 0;
  std::string name;
  Chunk* chunk;
  ~ObjFunction();
};

struct ObjInternedString : public Obj {
  constexpr static ObjType TYPE_ID = ObjType::OBJ_STRING;
  uint32_t hash;

  uint32_t static Hash(ObjInternedString* obj_str) { return obj_str->hash; }

  static ObjInternedString* Make(const char* data, int size);

  static ObjInternedString* Concat(const ObjInternedString* lhs, const ObjInternedString* rhs);

  Buffer<char> data;
  char* c_str() { return data.data(); }
  [[nodiscard]] const char* c_str() const { return data.data(); }
  int size() const;
  ~ObjInternedString();

 protected:
  struct InternView {
    const char* data;
    uint32_t hash;
    int size;
    bool operator==(const InternView& rhs) const {
      return size == rhs.size && hash == rhs.hash && strncmp(data, rhs.data, size) == 0;
      ;
    }
    bool operator!=(const InternView& rhs) const { return !(*this == rhs); }
  };
  InternView GetInternView() { return {.data = c_str(), .hash = hash, .size = size()}; }
  static uint32_t CachedStringHash(InternView string) { return string.hash; }

  using InternMap = HashMap<InternView, ObjInternedString*, CachedStringHash>;
  static InternMap& GetInternMap();
  uint32_t UpdateHash();
  ObjInternedString(const char* buf, int size);
  explicit ObjInternedString(Buffer<char>&& buffer);
  ObjInternedString() = delete;
  void TryInternThis();
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
