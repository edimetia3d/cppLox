//
// LICENSE: MIT
//

#include "clox_object.h"

#include <cstdio>

namespace lox {
namespace vm {
bool lox::vm::Obj::Equal(const lox::vm::Obj* rhs) const {
  if (type != rhs->type) {
    return false;
  }
  switch (type) {
    case ObjType::OBJ_STRING:
      return As<ObjString>()->data == rhs->As<ObjString>()->data;
  }
}
void Obj::Print() const {
#define q_printf(...)  \
  printf(__VA_ARGS__); \
  break
  switch (type) {
    case ObjType::OBJ_STRING:
      q_printf("%s", As<ObjString>()->c_str());
  }
#undef q_printf
}
}  // namespace vm
}  // namespace lox
