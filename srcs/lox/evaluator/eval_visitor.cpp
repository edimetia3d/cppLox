//
// LICENSE: MIT
//

#include "eval_visitor.h"

namespace lox {

object::LoxObject AstEvaluator::Eval(const Expr& expr) const { return expr.Accept(*this); }
object::LoxObject AstEvaluator::VisitLiteral(const Literal& expr) const {
  switch (expr.value.type_) {
    case TokenType::NUMBER:
      return object::LoxObject(std::stod(expr.value.lexeme_));
    case TokenType::STRING:
      return object::LoxObject(expr.value.lexeme_);
    default:
      throw "Not Valid Literal";
  }
}
object::LoxObject AstEvaluator::VisitGrouping(const Grouping& expr) const { return Eval(expr.expression); }
object::LoxObject AstEvaluator::VisitUnary(const Unary& expr) const {
  auto right = Eval(expr.right);

  switch (expr.op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      return !right;
    default:
      throw "Not supported unary";
  }
}
object::LoxObject AstEvaluator::VisitBinary(const Binary& expr) const { return Visitor::VisitBinary(expr); }
}  // namespace lox
