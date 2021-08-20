//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_

#include <string>

#include "lox/ast/stmt.h"
#include "lox/visitors/evaluator/lox_function.h"
namespace lox {
class LoxClass;
struct LoxClassData {
  std::string name;
  std::map<std::string, object::LoxObject> methods;
  std::shared_ptr<LoxClass> superclass;
};
class LoxClass : public LoxCallable {
 public:
  using RawValueT = LoxClassData;
  int Arity() override;
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) override;
  std::string ToString() const override;

  object::LoxObject GetMethod(std::string name) {
    if (Data().methods.count(name) != 0) {
      return Data().methods[name];
    }
    if (Data().superclass) {
      return Data().superclass->GetMethod(name);
    }
    return object::VoidObject();
  }

 private:
  RawValueT& Data() { return AsNative<RawValueT>(); }
  const RawValueT& Data() const { return AsNative<RawValueT>(); }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(LoxClass);
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
