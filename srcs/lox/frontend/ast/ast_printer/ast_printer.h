//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
#define CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_

#include <cassert>
#include <string>

#include "lox/frontend/ast/ast.h"

namespace lox {
class AstPrinter : public AstNodeVisitor<std::string> {
 public:
  std::string Print(std::shared_ptr<AstNode> node) {
    assert(IsValid(node));
    node->Accept(this);
    return PopRet();
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
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
