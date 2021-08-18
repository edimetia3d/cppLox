//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_

#include "lox/lox_object/lox_object_state.h"
#include "lox/visitors/evaluator/lox_class.h"
namespace lox {
class LoxClassInstanceState : public object::LoxObjectState {
 public:
  LoxClassInstanceState(LoxClassState* klass) : klass_(klass) {}

  std::string ToString() override;

 private:
  LoxClassState* klass_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
