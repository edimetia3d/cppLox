//
// LICENSE: MIT
//

#include "semantic_check.h"

#include "lox/common/lox_error.h"
namespace lox {

void SemanticCheck::PreNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) {
  if (auto p = CastTo<WhileStmt>(ast_node)) {
    ++while_loop_level;
    return;
  }
  if (auto p = CastTo<BreakStmt>(ast_node)) {
    if (while_loop_level == 0) {
      throw LoxError("Semantic Error: " + p->src_token()->Dump() + " Nothing to break here.");
    }
    return;
  }
  if (auto p = CastTo<ClassStmt>(ast_node)) {
    if (IsValid(p->superclass())) {
      if (p->superclass()->DownCast<VariableExpr>()->name()->lexeme == p->name()->lexeme) {
        throw LoxError("Semantic Error: " + p->name()->Dump() + " Class can not inherit itself");
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
