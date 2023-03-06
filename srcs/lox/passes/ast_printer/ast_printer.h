//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
#define CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_

#include <cassert>
#include <string>

#include "lox/ast/ast.h"

namespace lox {
class AstPrinter : public ASTNodeVisitor<std::string> {
public:
  std::string Print(ASTNode *node) {
    assert(node);
    return ValueVisit(node);
  }

protected:
  void Visit(LogicalExpr *node) override;
  void Visit(BinaryExpr *node) override;
  void Visit(GroupingExpr *node) override;
  void Visit(LiteralExpr *node) override;
  void Visit(UnaryExpr *node) override;
  void Visit(VariableExpr *node) override;
  void Visit(AssignExpr *node) override;
  void Visit(CallExpr *node) override;
  void Visit(GetAttrExpr *node) override;
  void Visit(SetAttrExpr *node) override;
  void Visit(PrintStmt *node) override;
  void Visit(ReturnStmt *node) override;
  void Visit(WhileStmt *node) override;
  void Visit(ForStmt *node) override;
  void Visit(BreakStmt *node) override;
  void Visit(ExprStmt *node) override;
  void Visit(VarDeclStmt *node) override;
  void Visit(FunctionStmt *node) override;
  void Visit(ClassStmt *node) override;
  void Visit(BlockStmt *node) override;
  void Visit(IfStmt *node) override;
  void Visit(CommaExpr *node) override;
  void Visit(ListExpr *node) override;
  void Visit(GetItemExpr *node) override;
  void Visit(TensorExpr *node) override;

  int semantic_level = 0;
  struct SemanticLevelGuard {
    SemanticLevelGuard(AstPrinter *printer_) : printer(printer_) { ++printer->semantic_level; }
    ~SemanticLevelGuard() { --printer->semantic_level; }
    AstPrinter *printer;
  };
  std::string Indentation() {
    std::vector<char> buf(semantic_level * 2 + 1, ' ');
    buf.back() = '\0';
    return buf.data();
  }
  std::string &PossibleBlockPrint(ASTNode *node, std::string &ret);
};

} // namespace lox

#endif // CPPLOX_SRCS_LOX_AST_AST_PRINTER_H_
