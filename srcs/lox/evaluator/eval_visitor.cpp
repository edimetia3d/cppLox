//
// LICENSE: MIT
//

#include "eval_visitor.h"

namespace lox {

object::LoxObjectPointer AstEvaluator::Eval(const Expr& expr) const {
  return expr.Accept(*this);
}
object::LoxObjectPointer AstEvaluator::VisitLiteral(const Literal& expr) const {
  return object::LoxObject::FromLiteralToken(expr.value);
}
object::LoxObjectPointer AstEvaluator::VisitGrouping(
    const Grouping& expr) const {
  return Eval(expr.expression);
}
object::LoxObjectPointer AstEvaluator::VisitUnary(const Unary& expr) const {
  auto right = Eval(expr.right);

  switch (expr.op.type_) {
    case TokenType::MINUS:
      return -(*right);
    case TokenType::BANG:
      return !(*right);
    default:
      throw "Not supported unary";
  }
}
object::LoxObjectPointer AstEvaluator::VisitBinary(const Binary& expr) const {
  return Visitor::VisitBinary(expr);
}
}  // namespace lox
