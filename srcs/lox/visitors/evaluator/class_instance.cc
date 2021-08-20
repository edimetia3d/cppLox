//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstance::ToString() const { return std::string("Instance of ") + Data().klass->ToString(); }
object::LoxObject LoxClassInstance::GetAttr(std::string attr) {
  if (Data().dict.count(attr) != 0) {
    return Data().dict[attr];
  }
  auto ret = Data().klass->GetMethod(attr);
  if (IsValid(ret)) {
    ret = ret->DownCast<LoxFunction>()->BindThis(shared_from_this());
  }
  return ret;
}
void LoxClassInstance::SetAttr(std::string attr, object::LoxObject value) { Data().dict[attr] = value; }
}  // namespace lox
