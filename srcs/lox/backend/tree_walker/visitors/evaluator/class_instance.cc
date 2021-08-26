//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstance::ToString() const { return std::string("Instance of ") + Data().klass->ToString(); }
bool LoxClassInstance::IsInstanceOf(LoxClass *klass) {
  auto possible = Data().klass.get();
  while (possible && klass != possible) {
    possible = possible->RawValue<LoxClassData>().superclass.get();
  }
  return klass == possible;
}
}  // namespace lox
