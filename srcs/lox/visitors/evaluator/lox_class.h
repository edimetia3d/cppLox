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
  explicit LoxClassState(std::string name) : name_(name) {}
  int Arity() override;
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) override;
  std::string ToString() override;

 private:
  std::string name_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
