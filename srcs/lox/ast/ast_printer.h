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
  std::string Print(Expr expr) { return expr.Accept(this); }

 protected:
  std::string Visit(Binary* expr) override;
  std::string Visit(Grouping* expr) override;
  std::string Visit(Literal* expr) override;
  std::string Visit(Unary* expr) override;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
