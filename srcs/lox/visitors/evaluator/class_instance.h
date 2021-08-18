//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_

#include <map>

#include "lox/lox_object/lox_object_state.h"
#include "lox/visitors/evaluator/lox_class.h"
namespace lox {
class LoxClassInstanceState : public object::LoxObjectState {
 public:
  LoxClassInstanceState(LoxClassState* klass) : klass_(klass) {}

  std::string ToString() override;

  object::LoxObject GetAttr(std::string attr);

 private:
  LoxClassState* klass_;
  std::map<std::string, object::LoxObject> dict_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
