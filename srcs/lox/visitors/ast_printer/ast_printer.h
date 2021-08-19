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
class AstPrinter : public ExprVisitor, public StmtVisitor {
 public:
  std::string Print(Expr expr) {
    assert(IsValid(expr));
    return expr->Accept(this).AsNative<std::string>();
  }
  std::string Print(Stmt stmt) {
    assert(IsValid(stmt));
    return stmt->Accept(this).AsNative<std::string>();
  }

 protected:
  object::LoxObject Visit(LogicalExpr* state) override;
  object::LoxObject Visit(BinaryExpr* state) override;
  object::LoxObject Visit(GroupingExpr* state) override;
  object::LoxObject Visit(LiteralExpr* state) override;
  object::LoxObject Visit(UnaryExpr* state) override;
  object::LoxObject Visit(VariableExpr* state) override;
  object::LoxObject Visit(AssignExpr* state) override;
  object::LoxObject Visit(CallExpr* state) override;
  object::LoxObject Visit(GetAttrExpr* state) override;
  object::LoxObject Visit(SetAttrExpr* state) override;
  object::LoxObject Visit(PrintStmt* state) override;
  object::LoxObject Visit(ReturnStmt* state) override;
  object::LoxObject Visit(WhileStmt* state) override;
  object::LoxObject Visit(BreakStmt* state) override;
  object::LoxObject Visit(ExprStmt* state) override;
  object::LoxObject Visit(VarDeclStmt* state) override;
  object::LoxObject Visit(FunctionStmt* state) override;
  object::LoxObject Visit(ClassStmt* state) override;
  object::LoxObject Visit(BlockStmt* state) override;
  object::LoxObject Visit(IfStmt* state) override;
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
