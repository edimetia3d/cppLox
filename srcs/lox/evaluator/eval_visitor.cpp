//
// LICENSE: MIT
//

#include "eval_visitor.h"

namespace lox {

object::LoxObject AstEvaluator::VisitLiteral(Literal* expr) {
  switch (expr->value.type_) {
    case TokenType::NUMBER:
      return object::LoxObject(std::stod(expr->value.lexeme_));
    case TokenType::STRING:
      return object::LoxObject(expr->value.lexeme_);
    default:
      throw "Not Valid Literal";
  }
}
object::LoxObject AstEvaluator::VisitGrouping(Grouping* expr) { return Eval(expr->expression); }
object::LoxObject AstEvaluator::VisitUnary(Unary* expr) {
  auto right = Eval(expr->right);

  switch (expr->op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      return !right;
    default:
      throw "Not supported unary";
  }
}
object::LoxObject AstEvaluator::VisitBinary(Binary* expr) { return object::LoxObject((double)0); }
}  // namespace lox
