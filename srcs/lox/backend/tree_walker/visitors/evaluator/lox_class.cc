//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/visitors/evaluator/lox_class.h"

#include "lox/backend/tree_walker/visitors/evaluator/class_instance.h"

namespace lox {

int LoxClass::Arity() {
  auto init_fn = GetInitFn();
  if (IsValid(init_fn)) {
    return init_fn->Arity();
  }
  return 0;
}
object::LoxObject LoxClass::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  LoxClassInstanceData data{.klass = std::static_pointer_cast<LoxClass>(shared_from_this())};
  auto ret = object::MakeLoxObject<LoxClassInstance>(data);
  auto init_fn = GetInitFn();
  if (IsValid(init_fn)) {
    auto bounded_initializer = init_fn->BindThis(ret);
    bounded_initializer->DownCast<LoxCallable>()->Call(evaluator, arguments);
  }
  return ret;
}
std::string LoxClass::ToString() const { return std::string("class ") + Data().name; }
std::shared_ptr<LoxFunction> LoxClass::GetInitFn() {
  auto initializer = object::LoxObject();
  if (dict.contains("init")) {
    initializer = dict["init"];
  }
  if (auto init_fn = initializer->DownCast<LoxFunction>()) {
    return std::static_pointer_cast<LoxFunction>(init_fn->shared_from_this());
  }
  return nullptr;
}
}  // namespace lox
