//
// LICENSE: MIT
//

#include "lox/frontend/passes/semantic_check/semantic_check.h"

#include "lox/lox_error.h"
namespace lox {

void SemanticCheck::PreNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) {
  if (auto p = CastTo<WhileStmt>(ast_node)) {
    ++while_loop_level;
    return;
  }
  if (auto p = CastTo<BreakStmt>(ast_node)) {
    if (while_loop_level == 0) {
      throw LoxError("Semantic Error: " + p->src_token()->Str() + " Nothing to break here.");
    }
    return;
  }
  if (auto p = CastTo<ClassStmt>(ast_node)) {
    if (IsValid(p->superclass())) {
      if (p->superclass()->DownCast<VariableExpr>()->name()->lexeme == p->name()->lexeme) {
        throw LoxError("Semantic Error: " + p->name()->Str() + " Class can not inherit itself");
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
