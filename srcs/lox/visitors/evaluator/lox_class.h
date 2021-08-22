//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_

#include <string>

#include "lox/ast/ast.h"
#include "lox/visitors/evaluator/lox_function.h"
namespace lox {
class LoxClass;
struct LoxClassData {
  std::string name;
  std::shared_ptr<LoxClass> superclass;
};
class LoxClass : public LoxCallable {
 public:
  using RawValueT = LoxClassData;
  int Arity() override;
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) override;
  std::string ToString() const override;

  object::LoxObject GetAttr(const std::string& name) override {
    if (dict.contains(name)) {
      return dict[name];
    }
    if (Data().superclass) {
      return Data().superclass->GetAttr(name);
    }
    return object::VoidObject();
  }

 private:
  std::shared_ptr<LoxFunction> GetInitFn();
  RawValueT& Data() { return RawValue<RawValueT>(); }
  const RawValueT& Data() const { return RawValue<RawValueT>(); }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(LoxClass);
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
