//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
#define CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_

#include <string>

#include "lox/ast/ast_printer.h"
#include "lox/ast/expr.h"

namespace lox {
class AstPrinter : public Visitor<std::string> {
 public:
  std::string Print(ExprPointer p_expr) const { return Print(*p_expr); }

  std::string Print(const Expr& expr) const;

 protected:
  std::string VisitBinary(const Binary& expr) const override;
  std::string VisitGrouping(const Grouping& expr) const override;
  std::string VisitLiteral(const Literal& expr) const override;
  std::string VisitUnary(const Unary& expr) const override;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
