//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_class.h"

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

int LoxClassState::Arity() {
  auto initializer = GetMethod("init");
  if (initializer.IsValid()) {
    return initializer.DownCastState<LoxFunctionState>()->Arity();
  }
  return 0;
}
object::LoxObject LoxClassState::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  auto ret = object::LoxObject(new LoxClassInstanceState(this));
  auto initializer = GetMethod("init");
  if (initializer.IsValid()) {
    auto init_fn = initializer.DownCastState<LoxFunctionState>()->BindThis(ret);
    init_fn.DownCastState<LoxCallableState>()->Call(evaluator, arguments);
  }
  return ret;
}
std::string LoxClassState::ToString() { return std::string("class ") + name_; }
}  // namespace lox
