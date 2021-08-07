//
// License: MIT
//
#include "lox/ast/ast_printer.h"

namespace lox {

object::LoxObject lox::ExprPrinter::Visit(BinaryState* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("(") + left_expr + op + right_expr + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(LiteralState* state) { return object::LoxObject(state->value.lexeme_); }
object::LoxObject lox::ExprPrinter::Visit(GroupingState* state) {
  return object::LoxObject(std::string("(") + Print(state->expression) + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(UnaryState* state) {
  return object::LoxObject(std::string("(") + state->op.lexeme_ + Print(state->right) + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(VariableState* state) {
  return object::LoxObject(std::string("Var{") + state->name.lexeme_ + std::string("}"));
}

}  // namespace lox
