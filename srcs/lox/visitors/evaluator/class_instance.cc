//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstance::ToString() const { return std::string("Instance of ") + Data().klass->ToString(); }
object::LoxObject LoxClassInstance::GetAttr(const std::string& attr) {
  if (dict.contains(attr)) {
    return dict[attr];
  }
  auto ret = Data().klass->GetAttr(attr);
  if (IsValid(ret)) {
    if (auto function = ret->DownCast<LoxFunction>()) {
      ret = function->BindThis(shared_from_this());
    }
  }
  return ret;
}
}  // namespace lox
