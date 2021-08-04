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

class LoxObjectOperator;

class LoxObject {
 public:
  template <class RealT>
  explicit LoxObject(LoxObjectOperator* lox_oprator, const RealT& value)
      : lox_operator(lox_oprator), raw_value(new RealT{value}){};
  template <class RealT>
  explicit LoxObject(std::shared_ptr<LoxObjectOperator> lox_operator, const RealT& value)
      : lox_operator(std::move(lox_operator)), raw_value(new RealT{value}) {}

  explicit LoxObject(bool);
  explicit LoxObject(double);
  explicit LoxObject(char* v) : LoxObject(std::string(v)){};
  explicit LoxObject(const std::string&);
  // Uary
  LoxObject operator-();
  LoxObject operator!();

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
  operator bool() const;

  template <class T>
  T& AsNative() {
    return *static_cast<T*>(raw_value.get());
  }

  template <class T>
  const T& AsNative() const {
    return *static_cast<T*>(raw_value.get());
  }

 private:
  static void BinaryTypeCheck(LoxObject* lhs, LoxObject* rhs);

  std::shared_ptr<LoxObjectOperator> lox_operator;
  std::shared_ptr<void> raw_value;
};

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
