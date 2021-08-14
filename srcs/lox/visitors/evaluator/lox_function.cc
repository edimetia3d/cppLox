//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_function.h"

namespace lox {

int LoxFunctionState::Arity() { return function->params.size(); }
LoxFunctionState::LoxFunctionState(const FunctionStmtState *state)
    : function(new FunctionStmtState(state->name, state->params, state->body)) {}
object::LoxObject LoxFunctionState::Call(StmtEvaluator *evaluator, std::vector<object::LoxObject> arguments) {
  StmtEvaluator::EnterNewScopeGuard guard(evaluator);
  int N = function->params.size();
  for (int i = 0; i < N; ++i) {
    evaluator->WorkEnv()->Define(function->params[i].lexeme_, arguments[i]);
  }
  for (auto &stmt : function->body) {
    evaluator->Eval(stmt);
  }
  // todo: support return value
  return object::LoxObject(0.0);
}
std::string LoxFunctionState::ToString() { return std::string("Function ") + function->name.lexeme_; }
}  // namespace lox
