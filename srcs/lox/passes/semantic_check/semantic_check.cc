//
// Created by edi on 8/21/21.
//

#include "lox/passes/semantic_check/semantic_check.h"

#include "lox/error.h"
namespace lox {

void SemanticCheck::PreNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) {
  if (auto p = CastTo<WhileStmt>(ast_node)) {
    ++while_loop_level;
    return;
  }
  if (auto p = CastTo<BreakStmt>(ast_node)) {
    if (while_loop_level == 0) {
      throw SemanticError(Error(p->src_token(), "Nothing to break here."));
    }
    return;
  }
  if (auto p = CastTo<ClassStmt>(ast_node)) {
    if (IsValid(p->superclass())) {
      if (p->superclass()->DownCast<VariableExpr>()->name()->lexeme_ == p->name()->lexeme_) {
        throw SemanticError(Error(p->name(), "Class can not inherit itself"));
      }
    }
    return;
  }
}
void SemanticCheck::PostNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) {
  if (auto p = CastTo<WhileStmt>(ast_node)) {
    --while_loop_level;
    return;
  }
}
}  // namespace lox
