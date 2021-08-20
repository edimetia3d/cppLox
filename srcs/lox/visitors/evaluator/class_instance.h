//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_

#include <map>

#include "lox/lox_object/lox_object.h"
#include "lox/visitors/evaluator/lox_class.h"
namespace lox {
struct LoxClassInstanceData {
  std::shared_ptr<LoxClass> klass;
};
class LoxClassInstance : public object::LoxObjectBase {
 public:
  using RawValueT = LoxClassInstanceData;
  std::string ToString() const override;

  bool IsInstanceOf(LoxClass* klass);

 private:
  RawValueT& Data() { return RawValue<RawValueT>(); }
  const RawValueT& Data() const { return RawValue<RawValueT>(); }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(LoxClassInstance);
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_CLASS_INSTANCE_H_
