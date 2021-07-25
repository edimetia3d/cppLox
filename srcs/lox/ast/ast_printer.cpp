//
// License: MIT
//
#include "lox/ast/ast_printer.h"

namespace lox {

std::string lox::AstPrinter::Print(const lox::ExprImpl& expr) const { return expr.Accept(*this); }
std::string lox::AstPrinter::VisitBinary(const Binary& binary_expr) const {
  std::string left_expr = Print(binary_expr.left);
  std::string op = binary_expr.op.lexeme_;
  std::string right_expr = Print(binary_expr.right);
  return std::string("(") + left_expr + op + right_expr + std::string(")");
}
std::string AstPrinter::VisitLiteral(const Literal& expr) const {
  return expr.value.lexeme_;
}
std::string lox::AstPrinter::VisitGrouping(const Grouping& expr) const {
  return std::string("(") + Print(expr.expression) + std::string(")");
}
std::string AstPrinter::VisitUnary(const Unary& expr) const {
  return std::string("(") + expr.op.lexeme_ + Print(expr.right) +
         std::string(")");
}

}  // namespace lox
