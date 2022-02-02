//
// LICENSE: MIT
//

#include "value.h"

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
      char buf[20];
      snprintf(buf, 20, "%g", AsNumber());
      return buf;
    }
    case ValueType::NIL:
      return "nil";
    case ValueType::OBJECT:
      return AsObject()->Str();
    default:
      return "UnKownValue";
  }
}
bool Value::IsTrue() const { return !IsNil() && (!IsBool() || AsBool()); }
}  // namespace lox
