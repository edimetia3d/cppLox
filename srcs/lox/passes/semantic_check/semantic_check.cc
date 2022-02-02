//
// LICENSE: MIT
//

#include "semantic_check.h"

#include "lox/common/lox_error.h"
namespace lox {

Pass::IsModified SemanticCheck::PreNode(ASTNode* ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>()) {
    ++while_loop_level;
    return Pass::IsModified::NO;
  }
  if (auto p = ast_node->DynAs<BreakStmt>()) {
    if (while_loop_level == 0) {
      throw LoxError("Semantic Error: " + p->attr->src_token->Dump() + " Nothing to break here.");
    }
    return Pass::IsModified::NO;
  }
  if (auto p = ast_node->DynAs<ClassStmt>()) {
    if (p->superclass) {
      if (p->superclass->DynAs<VariableExpr>()->attr->name->lexeme == p->attr->name->lexeme) {
        throw LoxError("Semantic Error: " + p->attr->name->Dump() + " Class can not inherit itself");
      }
    }
    return Pass::IsModified::NO;
  }
  if (auto p = ast_node->DynAs<FunctionStmt>()) {
    if (p->attr->params.size() >= 255) {
      throw LoxError("Lox Can't have more than 255 arguments.");
    }
  }
  if (auto p = ast_node->DynAs<CallExpr>()) {
    if (p->arguments.size() >= 255) {
      throw LoxError("Lox Can't have more than 255 arguments.");
    }
  }

  // todo: check init must return nothing

  return Pass::IsModified::NO;
}
Pass::IsModified SemanticCheck::PostNode(ASTNode* ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>()) {
    --while_loop_level;
  }
  return Pass::IsModified::NO;
}
}  // namespace lox
