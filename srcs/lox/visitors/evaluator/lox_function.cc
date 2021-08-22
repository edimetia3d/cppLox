//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_function.h"

namespace lox {

int LoxFunction::Arity() { return Data().function->params.size(); }
object::LoxObject LoxFunction::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  Evaluator::EnterNewScopeGuard guard(evaluator, Data().closure);
  int N = Data().function->params.size();
  for (int i = 0; i < N; ++i) {
    evaluator->WorkEnv()->Define(Data().function->params[i]->lexeme_, arguments[i]);
  }
  try {
    for (auto &stmt : Data().function->body) {
      evaluator->Eval(stmt);
    }
  } catch (ReturnValue &ret) {
    if (Data().is_init_method) {
      return evaluator->WorkEnv()->Get("this");
    }
    return ret.ret;
  }
  if (Data().is_init_method) {
    return evaluator->WorkEnv()->Get("this");
  }
  return object::VoidObject();
}
std::string LoxFunction::ToString() const { return std::string("Function ") + Data().function->name->lexeme_; }
object::LoxObject LoxFunction::BindThis(object::LoxObject obj_this) {
  auto new_env = Environment::Make(Data().closure);
  new_env->Define("this", obj_this);
  LoxFunctionData bound_init_data{
      .is_init_method = Data().is_init_method, .closure = new_env, .function = Data().function};
  return object::MakeLoxObject<LoxFunction>(bound_init_data);
}
}  // namespace lox
