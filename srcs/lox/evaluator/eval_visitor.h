//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include "lox/ast/expr.h"
#include "lox/evaluator/lox_object.h"
namespace lox {
class AstEvaluator : public Visitor<object::LoxObjectPointer> {
 public:
  object::LoxObjectPointer Eval(ExprPointer p_expr) const {
    return Eval(*p_expr);
  }

  object::LoxObjectPointer Eval(const Expr& expr) const;

 protected:
  object::LoxObjectPointer VisitBinary(const Binary& expr) const override;
  object::LoxObjectPointer VisitGrouping(const Grouping& expr) const override;
  object::LoxObjectPointer VisitLiteral(const Literal& expr) const override;
  object::LoxObjectPointer VisitUnary(const Unary& expr) const override;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
