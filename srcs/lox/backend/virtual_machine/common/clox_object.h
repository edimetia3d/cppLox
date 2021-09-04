//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
#include <cassert>

#include "lox/backend/virtual_machine/common/link_list.h"
#include "lox/backend/virtual_machine/common/vector.h"
namespace lox {
namespace vm {

#define CLOX_OBJ_HEAP_ONLY(CLASS_NAME) \
 protected:                            \
  CLASS_NAME() = default;              \
  friend struct Obj;

enum class ObjType { UNKNOWN, OBJ_STRING };

struct Obj {
  ObjType type;

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
  Obj() : type(ObjType::UNKNOWN) { AllCreatedObj().Insert(this); };
};

struct ObjString : public Obj {
  constexpr static ObjType TYPE_ID = ObjType::OBJ_STRING;

  static ObjString* Concat(const ObjString* lhs, const ObjString* rhs) {
    auto ret = new ObjString;
    ret->type = TYPE_ID;
    ret->data.reserve(lhs->size() + rhs->size());
    ret->data.push_buffer(lhs->c_str(), lhs->size());
    ret->data.push_buffer(rhs->c_str(), rhs->size());
    ret->data.push_back('\0');
    return ret;
  }
  Vector<char> data;
  char* c_str() { return data.data(); }
  [[nodiscard]] const char* c_str() const { return data.data(); }
  int size() const {
    return data.size() - 1;  // this is a c style str
  }
  CLOX_OBJ_HEAP_ONLY(ObjString);

 protected:
  ObjString(const char* buf, int size) {
    type = TYPE_ID;
    data.push_buffer(buf, size);
    data.push_back('\0');
  }
};

}  // namespace vm
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_CLOX_OBJECT_H_
