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
  object::LoxObject Eval(Expr p_expr) const { return Eval(*p_expr); }

  object::LoxObject Eval(const ExprImpl& expr) const;

 protected:
  object::LoxObject VisitBinary(const Binary& expr) const override;
  object::LoxObject VisitGrouping(const Grouping& expr) const override;
  object::LoxObject VisitLiteral(const Literal& expr) const override;
  object::LoxObject VisitUnary(const Unary& expr) const override;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
