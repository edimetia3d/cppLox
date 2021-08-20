//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_class.h"

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

int LoxClass::Arity() {
  auto initializer = GetMethod("init");
  if (IsValid(initializer)) {
    return initializer->DownCast<LoxFunction>()->Arity();
  }
  return 0;
}
object::LoxObject LoxClass::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  LoxClassInstanceData data{.klass = std::static_pointer_cast<LoxClass>(shared_from_this())};
  auto ret = object::MakeLoxObject<LoxClassInstance>(data);
  auto initializer = GetMethod("init");
  if (IsValid(initializer)) {
    auto bounded_initializer = initializer->DownCast<LoxFunction>()->BindThis(ret);
    bounded_initializer->DownCast<LoxCallable>()->Call(evaluator, arguments);
  }
  return ret;
}
std::string LoxClass::ToString() const { return std::string("class ") + Data().name; }
}  // namespace lox
