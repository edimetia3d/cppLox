//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstance::ToString() const { return std::string("Instance of ") + Data().klass->ToString(); }
}  // namespace lox
