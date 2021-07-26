//
// License: MIT
//
#include "lox/ast/ast_printer.h"

namespace lox {

std::string lox::AstPrinter::Visit(BinaryState* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return std::string("(") + left_expr + op + right_expr + std::string(")");
}
std::string AstPrinter::Visit(LiteralState* state) { return state->value.lexeme_; }
std::string lox::AstPrinter::Visit(GroupingState* state) {
  return std::string("(") + Print(state->expression) + std::string(")");
}
std::string AstPrinter::Visit(UnaryState* state) {
  return std::string("(") + state->op.lexeme_ + Print(state->right) + std::string(")");
}

}  // namespace lox
