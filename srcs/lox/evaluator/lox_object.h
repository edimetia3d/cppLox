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

class LoxObjectImpl;
using LoxObjectImplPointer = std::shared_ptr<LoxObjectImpl>;

class LoxObject {
 public:
  explicit LoxObject(LoxObjectImplPointer impl) : impl(std::move(impl)) {}
  explicit LoxObject(bool);
  explicit LoxObject(double);
  explicit LoxObject(char* v) : LoxObject(std::string(v)){};
  explicit LoxObject(const std::string&);
  LoxObject operator-();
  LoxObject operator!();
  std::string ToString();
  operator bool() const;

 private:
  LoxObjectImplPointer impl;
};

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
