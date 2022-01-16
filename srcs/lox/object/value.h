//
// LICENSE: MIT
//

#ifndef CLOX_SRCS_CLOX_CLOX_VALUE_H_
#define CLOX_SRCS_CLOX_CLOX_VALUE_H_

#include <cassert>
#include "lox/object/object.h"

namespace lox {

enum class ValueType { NIL, NUMBER, BOOL, OBJECT };
struct Value {
  Value() : type(ValueType::NIL), as{.number = 0} {};
  explicit Value(double number) : type(ValueType::NUMBER), as{.number = number} {}
  explicit Value(bool boolean) : type(ValueType::BOOL), as{.boolean = boolean} {}
  explicit Value(Object* obj) : type(ValueType::OBJECT), as{.object = obj} {}
  [[nodiscard]] bool AsBool() const {
    assert(IsBool());
    return as.boolean;
  };
  [[nodiscard]] double AsNumber() const {
    assert(IsNumber());
    return as.number;
  };
  [[nodiscard]] const Object* AsObject() const {
    assert(IsObject());
    return as.object;
  }
  Object* AsObject() {
    assert(IsObject());
    return as.object;
  }

  [[nodiscard]] bool IsNil() const { return type == ValueType::NIL; }
  [[nodiscard]] bool IsBool() const { return type == ValueType::BOOL; }
  [[nodiscard]] bool IsNumber() const { return type == ValueType::NUMBER; }
  [[nodiscard]] bool IsObject() const { return type == ValueType::OBJECT; }
  [[nodiscard]] ValueType Type() const { return type; }
  bool Equal(Value rhs);
  std::string Str() const;

  [[nodiscard]] bool IsTrue() const { return !IsNil() && IsBool() && AsBool(); }

 private:
  ValueType type;
  union {
    bool boolean;
    double number;
    Object* object;
  } as;
};

}  // namespace lox
#endif  // CLOX_SRCS_CLOX_CLOX_VALUE_H_
