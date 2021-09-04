//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#include <cassert>

#include "lox/backend/virtual_machine/common/buffer.h"
#include "lox/backend/virtual_machine/common/link_list.h"
namespace lox {
namespace vm {


enum class ObjType { UNKNOWN, OBJ_STRING };

struct Obj {
  ObjType type;
  uint32_t hash;

  template <class T, class... Args>
  static T* Make(Args... args) {
    auto ret = new T(args...);
    return ret;
  }

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
  Obj(uint32_t hash) : type(ObjType::UNKNOWN), hash(hash) { AllCreatedObj().Insert(this); };
};

struct ObjString : public Obj {
  constexpr static ObjType TYPE_ID = ObjType::OBJ_STRING;

  static ObjString* Concat(const ObjString* lhs, const ObjString* rhs) {
    auto ret = new ObjString(lhs->c_str(), lhs->size());
    ret->Append(rhs);
    return ret;
  }
  void Append(const ObjString* rhs) {
    data.resize(data.size() - 1);
    data.push_buffer(rhs->c_str(), rhs->size());
    data.push_back('\0');
    UpdateHash();
  }
  Buffer<char> data;
  char* c_str() { return data.data(); }
  [[nodiscard]] const char* c_str() const { return data.data(); }
  int size() const {
    return data.size() - 1;  // this is a c style str
  }

 protected:
  uint32_t UpdateHash();
  ObjString(const char* buf, int size) : Obj(0) {
    type = TYPE_ID;
    data.push_buffer(buf, size);
    data.push_back('\0');
    UpdateHash();
  }
  ObjString() = delete;
  friend Obj;
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
