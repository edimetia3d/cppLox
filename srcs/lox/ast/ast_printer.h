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
  std::string Print(Expr expr) { return Dispatch(expr.State().get()); }

 protected:
  std::string Visit(BinaryState* state) override;
  std::string Visit(GroupingState* state) override;
  std::string Visit(LiteralState* state) override;
  std::string Visit(UnaryState* state) override;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
