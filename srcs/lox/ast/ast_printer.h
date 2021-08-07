//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
#define CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_

#include <cassert>
#include <string>

#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"

namespace lox {
class ExprPrinter : public ExprVisitor {
 public:
  std::string Print(Expr expr) {
    assert(expr.State());
    return expr.State()->Accept(this).AsNative<std::string>();
  }

 protected:
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
};

class StmtPrinter : public StmtVisitor {
 public:
  std::string Print(Stmt stmt) {
    assert(stmt.State());
    return stmt.State()->Accept(this).AsNative<std::string>();
  }

 protected:
  object::LoxObject Visit(PrintState* state) override;
  object::LoxObject Visit(ExpressionState* state) override;
  object::LoxObject Visit(VarState* state) override;

 private:
  ExprPrinter expr_printer_;
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
