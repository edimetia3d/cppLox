//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstanceState::ToString() { return std::string("Instance of ") + klass_->ToString(); }
object::LoxObject LoxClassInstanceState::GetAttr(std::string attr) {
  if (dict_.count(attr) != 0) {
    return dict_[attr];
  }
  auto ret = klass_->GetMethod(attr);
  if (ret.IsValid()) {
    ret = ret.DownCastState<LoxFunctionState>()->BindThis(object::LoxObject(shared_from_this()));
  }
  return ret;
}
void LoxClassInstanceState::SetAttr(std::string attr, object::LoxObject value) { dict_[attr] = value; }
}  // namespace lox
