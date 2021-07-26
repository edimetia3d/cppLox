//
// LICENSE: MIT
//

#include "eval_visitor.h"

namespace lox {

object::LoxObject AstEvaluator::Visit(LiteralState* state) {
  switch (state->value.type_) {
    case TokenType::NUMBER:
      return object::LoxObject(std::stod(state->value.lexeme_));
    case TokenType::STRING:
      return object::LoxObject(state->value.lexeme_);
    default:
      throw "Not Valid Literal";
  }
}
object::LoxObject AstEvaluator::Visit(GroupingState* state) { return Eval(state->expression); }
object::LoxObject AstEvaluator::Visit(UnaryState* state) {
  auto right = Eval(state->right);

  switch (state->op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      return !right;
    default:
      throw "Not supported unary";
  }
}
object::LoxObject AstEvaluator::Visit(BinaryState* state) { return object::LoxObject((double)0); }
}  // namespace lox
