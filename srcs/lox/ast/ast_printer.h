//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
#define CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_

#include <string>

#include "lox/ast/ast_printer.h"
#include "lox/ast/expr.h"

namespace lox {
class AstPrinter : public ExprVisitor {
 public:
  std::string Print(Expr expr) { return expr.State()->Accept(this).AsNative<std::string>(); }

 protected:
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
