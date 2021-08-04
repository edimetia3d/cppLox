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
    case TokenType::TRUE:
      return object::LoxObject(true);
    case TokenType::FALSE:
      return object::LoxObject(false);
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
object::LoxObject AstEvaluator::Visit(BinaryState* state) {
  auto left = Eval(state->left);
  auto right = Eval(state->right);

  switch (state->op.type_) {
    case TokenType::PLUS:
      return left + right;
    case TokenType::MINUS:
      return left - right;
    case TokenType::STAR:
      return left * right;
    case TokenType::SLASH:
      return left / right;
    case TokenType::EQUAL_EQUAL:
      return left == right;
    case TokenType::BANG_EQUAL:
      return left != right;
    case TokenType::LESS:
      return left < right;
    case TokenType::GREATER:
      return left > right;
    case TokenType::LESS_EQUAL:
      return left <= right;
    case TokenType::GREATER_EQUAL:
      return left >= right;
    default:
      throw "Not supported unary";
  }
}
}  // namespace lox
