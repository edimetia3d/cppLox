//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_

#include "lox/backend/tree_walker/evaluator/callable_object.h"
#include "lox/backend/tree_walker/evaluator/environment.h"
namespace lox {
struct LoxFunctionData {
  bool is_init_method = false;
  std::shared_ptr<Environment> closure;
  std::shared_ptr<FunctionStmt> function;
};
class LoxFunction : public LoxCallable {
 public:
  using RawValueT = LoxFunctionData;

  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) override;

  int Arity() override;

  std::string ToString() const override;

  object::LoxObject BindThis(object::LoxObject obj_this);

 private:
  RawValueT& Data() { return RawValue<RawValueT>(); }
  const RawValueT& Data() const { return RawValue<RawValueT>(); }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(LoxFunction);
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_FUNCTION_H_
