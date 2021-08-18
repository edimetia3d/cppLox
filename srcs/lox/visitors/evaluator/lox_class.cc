//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/lox_class.h"

#include "lox/visitors/evaluator/class_instance.h"

namespace lox {

int LoxClassState::Arity() { return 0; }
object::LoxObject LoxClassState::Call(Evaluator *evaluator, std::vector<object::LoxObject> arguments) {
  return object::LoxObject(new LoxClassInstanceState(this));
}
std::string LoxClassState::ToString() { return std::string("class ") + name_; }
}  // namespace lox
