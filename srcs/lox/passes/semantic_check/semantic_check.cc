//
// LICENSE: MIT
//

#include "semantic_check.h"

namespace lox {

Pass::IsModified SemanticCheck::PreNode(ASTNode* ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>() || ast_node->DynAs<ForStmt>()) {
    loop_infos.push_back(LoopInfo{});
    return Pass::IsModified::NO;
  }

  if (auto p = ast_node->DynAs<BreakStmt>()) {
    if (loop_infos.empty()) {
      throw SemanticError("Semantic Error: " + p->attr->src_token->Dump() + " Nothing to break here.");
    }
    return Pass::IsModified::NO;
  }

  if (auto p = ast_node->DynAs<ClassStmt>()) {
    all_classes[p->attr->name->lexeme] = std::make_shared<ClassInfo>();
    class_infos.push_back(all_classes[p->attr->name->lexeme]);
    if (p->superclass) {
      if (p->superclass->DynAs<VariableExpr>()->attr->name->lexeme == p->attr->name->lexeme) {
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Class can not inherit itself");
      }
      all_classes[p->attr->name->lexeme]->superclass =
          all_classes[p->superclass->DynAs<ClassStmt>()->attr->name->lexeme];
    }
    return Pass::IsModified::NO;
  }
  if (auto p = ast_node->DynAs<AssignExpr>()) {
    if (p->attr->name->lexeme == "this" || p->attr->name->lexeme == "super") {
      throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Can not assign to 'this/super'");
    }
  }

  if (auto p = ast_node->DynAs<VariableExpr>()) {
    if ((p->attr->name->lexeme == "super" || p->attr->name->lexeme == "this") && class_infos.empty()) {
      throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Can not use 'this/super' outside of class.");
    }
  }

  if (auto p = ast_node->DynAs<FunctionStmt>()) {
    if (p->attr->params.size() >= 255) {
      throw SemanticError("Lox Can't have more than 255 arguments.");
    }
  }
  if (auto p = ast_node->DynAs<CallExpr>()) {
    if (p->arguments.size() >= 255) {
      throw SemanticError("Lox Can't have more than 255 arguments.");
    }
  }

  // todo: check init must return nothing

  return Pass::IsModified::NO;
}
Pass::IsModified SemanticCheck::PostNode(ASTNode* ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>() || ast_node->DynAs<ForStmt>()) {
    loop_infos.pop_back();
    return Pass::IsModified::NO;
  }
  if (auto p = ast_node->DynAs<ClassStmt>()) {
    class_infos.pop_back();
    return Pass::IsModified::NO;
  }
  return Pass::IsModified::NO;
}
}  // namespace lox
