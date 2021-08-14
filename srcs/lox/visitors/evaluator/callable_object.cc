//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/callable_object.h"

#include "lox/lox_object/lox_object_state.h"
namespace lox {

class LoxCallableState : object::LoxObjectState {
 public:
  virtual int Arity() = 0;
  virtual object::LoxObject Call(ExprEvaluator* interpreter, std::vector<object::LoxObject> arguments) = 0;
};

object::LoxObject LoxCallable::Call(ExprEvaluator* interpreter, std::vector<object::LoxObject> arguments) {
  return ObjectState()->Call(interpreter, arguments);
}
int LoxCallable::Arity() { return ObjectState()->Arity(); }

LoxCallableState* LoxCallable::ObjectState() {
  auto ret = dynamic_cast<LoxCallableState*>(LoxObject::ObjectState());
  return ret;
}
bool LoxCallable::IsValid() {
  return LoxObject::IsValid() && dynamic_cast<LoxCallableState*>(LoxObject::ObjectState());
}

std::map<std::string, LoxCallable> BuiltinCallables() { return std::map<std::string, LoxCallable>(); }
}  // namespace lox
