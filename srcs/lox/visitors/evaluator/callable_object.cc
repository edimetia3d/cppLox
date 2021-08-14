//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/callable_object.h"

#include <time.h>

#include "lox/lox_object/lox_object_state.h"
namespace lox {

object::LoxObject LoxCallable::Call(StmtEvaluator* evaluator, std::vector<object::LoxObject> arguments) {
  return ObjectState()->Call(evaluator, arguments);
}
int LoxCallable::Arity() { return ObjectState()->Arity(); }

LoxCallableState* LoxCallable::ObjectState() { return dynamic_cast<LoxCallableState*>(LoxObject::ObjectState()); }
bool LoxCallable::IsValid() { return LoxObject::IsValid() && ObjectState() != nullptr; }

class Clock : public LoxCallableState {
 public:
  int Arity() override { return 0; }
  object::LoxObject Call(StmtEvaluator* evaluator, std::vector<object::LoxObject> arguments) {
    double clk = clock();
    object::LoxObjectStatePtr();
    return object::LoxObject(clk);
  }
  std::string ToString() override;
};
std::string Clock::ToString() { return "builtin function Clock"; }

const std::map<std::string, object::LoxObject>& BuiltinCallables() {
  static std::map<std::string, object::LoxObject> map{{"Clock", object::LoxObject(new Clock)}};
  return map;
};
}  // namespace lox
