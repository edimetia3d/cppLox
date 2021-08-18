//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

std::string LoxClassInstanceState::ToString() { return std::string("Instance of ") + klass_->ToString(); }
}  // namespace lox
