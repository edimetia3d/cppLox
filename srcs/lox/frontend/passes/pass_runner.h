//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#include <cassert>

#include "lox/frontend/ast/ast.h"
#include "pass.h"
namespace lox {
class PassRunner : public IAstNodeVisitor {
 public:
  PassRunner() = default;
  void SetPass(std::shared_ptr<Pass> pass) { pass_ = pass; }
  std::shared_ptr<AstNode> RunPass(std::shared_ptr<AstNode> node) {
    assert(IsValid(node));
    assert(pass_);
    std::shared_ptr<AstNode> new_node = node;
    ResetModify(node);
    pass_->PreNode(node.get(), &new_node);
    if (new_node != node || node->IsModified()) {
      node = RunPass(new_node);
    } else {
      node->Accept(this);
    }
    pass_->PostNode(node.get(), &new_node);
    if (new_node != node || node->IsModified()) {
      node = RunPass(new_node);
    }
    return node;
  }

 protected:
  void Visit(LogicalExpr* state) override;
  void Visit(BinaryExpr* state) override;
  void Visit(GroupingExpr* state) override;
  void Visit(LiteralExpr* state) override;
  void Visit(UnaryExpr* state) override;
  void Visit(VariableExpr* state) override;
  void Visit(AssignExpr* state) override;
  void Visit(CallExpr* state) override;
  void Visit(GetAttrExpr* state) override;
  void Visit(SetAttrExpr* state) override;

  void Visit(PrintStmt* state) override;
  void Visit(ReturnStmt* state) override;
  void Visit(WhileStmt* state) override;
  void Visit(BreakStmt* state) override;
  void Visit(ExprStmt* state) override;
  void Visit(VarDeclStmt* state) override;
  void Visit(FunctionStmt* state) override;
  void Visit(ClassStmt* state) override;
  void Visit(BlockStmt* state) override;
  void Visit(IfStmt* state) override;
  std::shared_ptr<Pass> pass_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
