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

class LoxObject;
using LoxObjectPointer = std::shared_ptr<LoxObject>;

class LoxObject {
 public:
  virtual ~LoxObject(){};

  virtual LoxObjectPointer operator-() = 0;
  virtual LoxObjectPointer operator!() = 0;
  virtual std::string ToString() = 0;
  static LoxObjectPointer FromLiteralToken(const Token& token);
};

class Number : public LoxObject {
 public:
  explicit Number(double value) : value(value) {}

  LoxObjectPointer operator-() override {
    return LoxObjectPointer(new Number(-value));
  }
  LoxObjectPointer operator!() override {
    return LoxObjectPointer(new Number(!static_cast<bool>(value)));
  }

  std::string ToString() override { return std::to_string(value); }

  double value;
};

class String : public LoxObject {
 public:
  explicit String(const std::string value) : value(value) {}
  LoxObjectPointer operator-() override { throw "Not supported"; }
  LoxObjectPointer operator!() override { throw "Not supported"; }
  std::string ToString() override { return value; }
  std::string value;
};

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
