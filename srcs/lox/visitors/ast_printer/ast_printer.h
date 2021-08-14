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
    assert(expr.IsValid());
    return expr.Accept(this).AsNative<std::string>();
  }

 protected:
  object::LoxObject Visit(LogicalState* state) override;
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
  object::LoxObject Visit(AssignState* state) override;
  object::LoxObject Visit(CallState* state) override;
};

class StmtPrinter : public StmtVisitor {
 public:
  std::string Print(Stmt stmt) {
    assert(stmt.IsValid());
    return stmt.Accept(this).AsNative<std::string>();
  }

 protected:
  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ReturnStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(FunctionStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;

 private:
  ExprPrinter expr_printer_;
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
