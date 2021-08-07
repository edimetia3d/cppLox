//
// LICENSE: MIT
//

#include "lox/ast/eval_visitor.h"

#include <iostream>

#include "lox/ast/ast_printer.h"
#include "lox/error.h"
namespace lox {

object::LoxObject ExprEvaluator::Visit(LiteralState* state) {
  switch (state->value.type_) {
    case TokenType::NUMBER:
      return object::LoxObject(std::stod(state->value.lexeme_));
    case TokenType::STRING:
      return object::LoxObject(std::string(state->value.lexeme_.begin() + 1, state->value.lexeme_.end() - 1));
    case TokenType::TRUE:
      return object::LoxObject(true);
    case TokenType::FALSE:
      return object::LoxObject(false);
    default:
      throw RuntimeError(Error(state->value, "Not a valid Literal."));
  }
}
object::LoxObject ExprEvaluator::Visit(GroupingState* state) { return Eval(state->expression); }
object::LoxObject ExprEvaluator::Visit(UnaryState* state) {
  auto right = Eval(state->right);

  switch (state->op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      return !right;
    default:
      throw RuntimeError(Error(state->op, "Not a valid Unary Op."));
  }
}
object::LoxObject ExprEvaluator::Visit(BinaryState* state) {
  auto left = Eval(state->left);
  auto right = Eval(state->right);
  try {
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
        throw RuntimeError(Error(state->op, "Not a valid Binary Op."));
    }
  } catch (const char* msg) {
    throw RuntimeError(Error(state->op, msg));
  }
}
object::LoxObject AstEvaluator::Visit(PrintState* state) {
  auto ret_v = expr_evaluator_.Eval(state->expression);
  static AstPrinter printer;
  std::cout << "Expr: " << printer.Print(state->expression) << std::endl;
  std::cout << "Str: " << ret_v.ToString() << std::endl;
  return object::LoxObject::VoidObject();
}
object::LoxObject AstEvaluator::Visit(ExpressionState* state) {
  expr_evaluator_.Eval(state->expression);
  return object::LoxObject::VoidObject();
}
}  // namespace lox
