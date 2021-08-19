//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_

#include <string>

#include "lox/ast/stmt.h"
#include "lox/visitors/evaluator/lox_function.h"
namespace lox {
class LoxClassState : public LoxCallableState {
 public:
  explicit LoxClassState(std::string name, std::map<std::string, object::LoxObject> methods)
      : name_(name), methods_(methods) {}
  int Arity() override;
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) override;
  std::string ToString() override;

  object::LoxObject GetMethod(std::string name) {
    if (methods_.count(name) != 0) {
      return methods_[name];
    }
    return object::LoxObject::VoidObject();
  }

 private:
  std::string name_;
  std::map<std::string, object::LoxObject> methods_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
