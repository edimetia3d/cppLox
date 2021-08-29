//
// LICENSE: MIT
//

#include "lox/backend/virtual_machine/common/clox_value.h"
namespace lox {

namespace vm {

void printValue(const Value &value) {
#define q_printf(...)  \
  printf(__VA_ARGS__); \
  break
  switch (value.Type()) {
    case ValueType::BOOL:
      q_printf((value.AsBool() ? "true" : "false"));
    case ValueType::NUMBER:
      q_printf("%f", value.AsNumber());
    case ValueType::NIL:
      q_printf("nil");
    default:
      q_printf("Unkown types");
  }
#undef q_printf
}
}  // namespace vm
}  // namespace lox
