//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_
#include <cassert>
#include <cstdio>

namespace lox {
namespace vm {
enum class ValueType { NIL, NUMBER, BOOL };
struct Value {
  Value() : type(ValueType::NIL), as{.number = 0} {};
  explicit Value(double number) : type(ValueType::NUMBER), as{.number = number} {}
  explicit Value(bool boolean) : type(ValueType::BOOL), as{.boolean = boolean} {}
  bool AsBool() const {
    assert(IsBool());
    return as.boolean;
  };
  double AsNumber() const {
    assert(IsNumber());
    return as.number;
  };
  bool IsNil() const { return type == ValueType::NIL; }
  bool IsBool() const { return type == ValueType::BOOL; }
  bool IsNumber() const { return type == ValueType::NUMBER; }
  ValueType Type() const { return type; }
  bool Equal(Value rhs) {
    if (type != rhs.type) return false;
    switch (type) {
      case ValueType::BOOL:
        return AsBool() == rhs.AsBool();
      case ValueType::NIL:
        return true;
      case ValueType::NUMBER:
        return AsNumber() == rhs.AsNumber();
      default:
        return false;  // Unreachable.
    }
  }

  bool IsTrue() { return !IsNil() && as.number; }

 private:
  ValueType type;
  union {
    bool boolean;
    double number;
  } as;
};
void printValue(const Value &value);
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_
