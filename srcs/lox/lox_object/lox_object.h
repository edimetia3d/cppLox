//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#include <memory>
#include <string>
#include <type_traits>

#include "lox/lox_object/lox_object_state.h"
#include "lox/token.h"
namespace lox {

namespace object {

template <class T>
concept SubclassOfLoxObjectState = std::is_base_of<LoxObjectState, T>::value;

class LoxObject {
 public:
  LoxObject() = default;
  explicit LoxObject(LoxObjectStatePtr ptr) : lox_object_state_(std::move(ptr)) {}
  explicit LoxObject(LoxObjectState* ptr) : lox_object_state_(ptr) {}
  explicit LoxObject(bool);
  explicit LoxObject(double);
  explicit LoxObject(char* v) : LoxObject(std::string(v)){};
  explicit LoxObject(const std::string&);
  static LoxObject VoidObject();
  // Uary
  LoxObject operator-();

  // Binary
  LoxObject operator-(LoxObject& rhs);
  LoxObject operator+(LoxObject& rhs);
  LoxObject operator*(LoxObject& rhs);
  LoxObject operator/(LoxObject& rhs);
  LoxObject operator==(LoxObject& rhs);
  LoxObject operator!=(LoxObject& rhs);
  LoxObject operator<(LoxObject& rhs);
  LoxObject operator>(LoxObject& rhs);
  LoxObject operator<=(LoxObject& rhs);
  LoxObject operator>=(LoxObject& rhs);
  std::string ToString();
  bool IsValid() { return static_cast<bool>(lox_object_state_); }
  bool IsValueTrue();

  template <class T>
  T& AsNative() {
    return *static_cast<T*>(RawObjPtr());
  }

  template <class T>
  const T& AsNative() const {
    return *static_cast<T*>(RawObjPtr());
  }

  template <SubclassOfLoxObjectState T>
  T* DownCastState() {
    return dynamic_cast<T*>(lox_object_state_.get());
  }

 private:
  void* RawObjPtr();
  void* RawObjPtr() const;
  LoxObjectStatePtr lox_object_state_;
};
}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
