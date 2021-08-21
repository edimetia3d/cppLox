//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#include <cassert>

#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
#include "lox/visitors/passes/pass.h"
namespace lox {
class PassRunner : public StmtVisitor, public ExprVisitor {
 public:
  PassRunner(std::shared_ptr<Pass> pass) : pass_(pass) {}
  void RunPass(Stmt stmt) {
    assert(IsValid(stmt));
    pass_->PreNode(stmt.get());
    stmt->Accept(this);
    pass_->PostNode(stmt.get());
  }

  void RunPass(Expr expr) {
    assert(IsValid(expr));
    pass_->PreNode(expr.get());
    expr->Accept(this);
    pass_->PostNode(expr.get());
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
  std::shared_ptr<Pass> pass_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
