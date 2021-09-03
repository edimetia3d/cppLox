//
// LICENSE: MIT
//

#include "clox_object.h"

#include <cstdio>

namespace lox {
namespace vm {
bool lox::vm::Obj::Equal(const lox::vm::Obj *rhs) const {
  if (type != rhs->type) {
    return false;
  }
  switch (type) {
    case ObjType::OBJ_STRING:
      return As<ObjString>()->data == rhs->As<ObjString>()->data;
    default:
      return false;
  }
}
void Obj::Print() const {
#define q_printf(...)  \
  printf(__VA_ARGS__); \
  break
  switch (type) {
    case ObjType::OBJ_STRING:
      q_printf("%s", As<ObjString>()->c_str());
    default:
      q_printf("Unknown Obj type");
  }
#undef q_printf
}
LinkList<Obj *> &Obj::AllCreatedObj() {
  static LinkList<Obj *> ret;
  return ret;
}
void Obj::Destroy(Obj *obj) {
  switch (obj->type) {
    case ObjType::OBJ_STRING:
      delete obj->As<ObjString>();
      break;
  }
}
}  // namespace vm
}  // namespace lox
