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
std::string Value::Str() const {
  switch (type) {
    case ValueType::BOOL:
      return AsBool() ? "true" : "false";
    case ValueType::NUMBER: {
      auto num = AsNumber();
      if (std::trunc(num) == num) {
        return std::to_string((int64_t)num);
      } else {
        return std::to_string(num);
      }
    }
    case ValueType::NIL:
      return "nil";
    case ValueType::OBJECT:
      return AsObject()->Str();
    default:
      return "UnKownValue";
  }
}
}  // namespace lox
