//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_
#include "lox/ast/stmt.h"
#include "lox/visitors/evaluator/callable_object.h"
#include "lox/visitors/evaluator/environment.h"
namespace lox {
class LoxFunctionState : public LoxCallableState {
 public:
  explicit LoxFunctionState(const FunctionStmt *state, std::shared_ptr<Environment> closure,
                            bool is_init_method = false);

  object::LoxObject Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) override;

  int Arity() override;

  std::string ToString() override;

  object::LoxObject BindThis(object::LoxObject obj_this);

 private:
  bool is_init_method = false;
  std::shared_ptr<Environment> closure;
  std::shared_ptr<FunctionStmt> function;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_
