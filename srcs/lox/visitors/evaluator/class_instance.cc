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
  return klass_->GetMethod(attr);
}
void LoxClassInstanceState::SetAttr(std::string attr, object::LoxObject value) { dict_[attr] = value; }
}  // namespace lox
