//
// LICENSE: MIT
//

#include "lox/object/value.h"

#include <numeric>

#include <cstdio>

namespace lox {

bool Value::Equal(Value rhs) {
  if (type != rhs.type) return false;
  switch (type) {
    case ValueType::BOOL:
      return AsBool() == rhs.AsBool();
    case ValueType::NIL:
      return true;
    case ValueType::NUMBER:
      return AsNumber() == rhs.AsNumber();
    case ValueType::OBJECT:
      return rhs.IsObject() && AsObject()->Equal(rhs.AsObject());
    default:
      return false;  // Unreachable.
  }
}
void Value::PrintLn() const {
  switch (type) {
    case ValueType::BOOL:
      printf(AsBool() ? "true\n" : "false\n");
      break;
    case ValueType::NUMBER: {
      auto num = AsNumber();
      if (std::trunc(num) == num) {
        printf("%d\n", (int)num);
      } else {
        printf("%f\n", num);
      }
      break;
    }
    case ValueType::NIL:
      printf("nil\n");
      break;
    case ValueType::OBJECT:
      printf("%s\n", AsObject()->Str().c_str());
      break;
    default:
      printf("UnKownValue");
  }
}
}  // namespace lox
