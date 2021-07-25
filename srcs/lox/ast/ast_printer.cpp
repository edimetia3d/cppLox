//
// License: MIT
//
#include "lox/ast/ast_printer.h"

namespace lox {

std::string lox::AstPrinter::Visit(Binary* expr) {
  std::string left_expr = Print(expr->left);
  std::string op = expr->op.lexeme_;
  std::string right_expr = Print(expr->right);
  return std::string("(") + left_expr + op + right_expr + std::string(")");
}
std::string AstPrinter::Visit(Literal* expr) { return expr->value.lexeme_; }
std::string lox::AstPrinter::Visit(Grouping* expr) {
  return std::string("(") + Print(expr->expression) + std::string(")");
}
std::string AstPrinter::Visit(Unary* expr) {
  return std::string("(") + expr->op.lexeme_ + Print(expr->right) + std::string(")");
}

}  // namespace lox
