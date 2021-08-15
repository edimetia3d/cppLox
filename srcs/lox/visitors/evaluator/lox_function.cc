//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_function.h"

namespace lox {

int LoxFunctionState::Arity() { return function->params.size(); }
LoxFunctionState::LoxFunctionState(const FunctionStmtState *state, std::shared_ptr<Environment> closure)
    : function(new FunctionStmtState(state->name, state->params, state->body)), closure(std::move(closure)) {}
object::LoxObject LoxFunctionState::Call(StmtEvaluator *evaluator, std::vector<object::LoxObject> arguments) {
  StmtEvaluator::EnterNewScopeGuard guard(evaluator, closure);
  int N = function->params.size();
  for (int i = 0; i < N; ++i) {
    evaluator->WorkEnv()->Define(function->params[i].lexeme_, arguments[i]);
  }
  for (auto &stmt : function->body) {
    evaluator->Eval(stmt);
  }
  return object::LoxObject::VoidObject();
}
std::string LoxFunctionState::ToString() { return std::string("Function ") + function->name.lexeme_; }
}  // namespace lox