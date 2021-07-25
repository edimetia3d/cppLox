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
  object::LoxObject Eval(Expr expr) { return expr.Accept(this); }

 protected:
  object::LoxObject VisitBinary(Binary* expr) override;
  object::LoxObject VisitGrouping(Grouping* expr) override;
  object::LoxObject VisitLiteral(Literal* expr) override;
  object::LoxObject VisitUnary(Unary* expr) override;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
