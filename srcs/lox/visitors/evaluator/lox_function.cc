//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_function.h"

namespace lox {

int LoxFunctionState::Arity() { return function->params.size(); }
LoxFunctionState::LoxFunctionState(const FunctionStmtState *state, std::shared_ptr<Environment> closure,
                                   bool is_init_method)
    : function(new FunctionStmtState(state->name, state->params, state->body)),
      closure(std::move(closure)),
      is_init_method(is_init_method) {}
object::LoxObject LoxFunctionState::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  Evaluator::EnterNewScopeGuard guard(evaluator, closure);
  int N = function->params.size();
  for (int i = 0; i < N; ++i) {
    evaluator->WorkEnv()->Define(function->params[i].lexeme_, arguments[i]);
  }
  try {
    for (auto &stmt : function->body) {
      evaluator->Eval(stmt);
    }
  } catch (ReturnValue &ret) {
    if (is_init_method) {
      return evaluator->WorkEnv()->Get("this");
    }
    return ret.ret;
  }
  if (is_init_method) {
    return evaluator->WorkEnv()->Get("this");
  }
  return object::LoxObject::VoidObject();
}
std::string LoxFunctionState::ToString() { return std::string("Function ") + function->name.lexeme_; }
object::LoxObject LoxFunctionState::BindThis(object::LoxObject obj_this) {
  auto new_env = Environment::Make(closure);
  new_env->Define("this", obj_this);
  return object::LoxObject(std::make_shared<LoxFunctionState>(function.get(), new_env, is_init_method));
}
}  // namespace lox
