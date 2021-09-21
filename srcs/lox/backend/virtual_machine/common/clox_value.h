//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_
#include <cassert>
#include <cstdio>

#include "lox/backend/virtual_machine/common/clox_object.h"
namespace lox {
namespace vm {
enum class ValueType { NIL, NUMBER, BOOL, OBJ };
struct Value {
  Value() : type(ValueType::NIL), as{.number = 0} {};
  explicit Value(double number) : type(ValueType::NUMBER), as{.number = number} {}
  explicit Value(bool boolean) : type(ValueType::BOOL), as{.boolean = boolean} {}
  explicit Value(Obj *obj) : type(ValueType::OBJ), as{.obj = obj} {}
  bool AsBool() const {
    assert(IsBool());
    return as.boolean;
  };
  double AsNumber() const {
    assert(IsNumber());
    return as.number;
  };
  const Obj *AsObj() const {
    assert(IsObj());
    return as.obj;
  }
  Obj *AsObj() {
    assert(IsObj());
    return as.obj;
  }
  bool IsNil() const { return type == ValueType::NIL; }
  bool IsBool() const { return type == ValueType::BOOL; }
  bool IsNumber() const { return type == ValueType::NUMBER; }
  bool IsObj() const { return type == ValueType::OBJ; }
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
      case ValueType::OBJ:
        return rhs.IsObj() && AsObj()->Equal(rhs.AsObj());
      default:
        return false;  // Unreachable.
    }
  }

  bool IsTrue() { return !IsNil() && IsBool() && AsBool(); }

 private:
  ValueType type;
  union {
    bool boolean;
    double number;
    Obj *obj;
  } as;
};
void printValue(const Value &value);
}  // namespace vm
}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_
