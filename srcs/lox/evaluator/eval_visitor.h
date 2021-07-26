//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include "lox/ast/expr.h"
#include "lox/evaluator/lox_object.h"
namespace lox {
class AstEvaluator : public Visitor<object::LoxObject> {
 public:
  object::LoxObject Eval(Expr expr) { return Dispatch(expr.State().get()); }

 protected:
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
