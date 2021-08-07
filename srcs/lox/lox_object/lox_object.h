//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#include <memory>
#include <string>

#include "lox/token.h"

namespace lox {

namespace object {

struct LoxObjectState;
using LoxObjectStatePtr = std::shared_ptr<LoxObjectState>;

class LoxObject {
 public:
  LoxObject() = default;
  LoxObject(LoxObjectStatePtr ptr) : lox_object_state_(ptr) {}
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

 private:
  void* RawObjPtr();
  void* RawObjPtr() const;
  LoxObjectStatePtr lox_object_state_;
};
}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
