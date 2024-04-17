//
// LICENSE: MIT
//

#include "semantic_check.h"

#include <cassert>

namespace lox {

Pass::IsModified SemanticCheck::PreNode(ASTNode *ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>() || ast_node->DynAs<ForStmt>()) {
    loop_infos.push_back(LoopInfo{});
  }

  if (auto p = ast_node->DynAs<BreakStmt>()) {
    if (loop_infos.empty()) {
      throw SemanticError("Semantic Error: " + p->attr->src_token->Dump() + " Nothing to break here.");
    }
  }
  if (auto p = ast_node->DynAs<VarDeclStmt>()) {
    auto name = p->attr->name->lexeme;
    if (p->initializer && p->initializer->DynAs<VariableExpr>()) {
      auto var_name = p->initializer->As<VariableExpr>()->attr->name->lexeme;
      if (var_name == name && function_infos.back().scopes.back().type != ScopeType::GLOBAL) {
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() +
                            " Can't read local variable in its own initializer.");
      }
    }
    AddNamedValue(name);
  }
  auto NO_VAR_FUNC_CLASS = [](ASTNode *p, const std::string &src) {
    if (!p) {
      return;
    }
    if (p->DynAs<VarDeclStmt>() || p->DynAs<FunctionStmt>() || p->DynAs<ClassStmt>()) {
      throw SemanticError("Semantic Error: Cannot declare variable/function/ inside " + src);
    }
  };
  if (auto p = ast_node->DynAs<IfStmt>()) {
    NO_VAR_FUNC_CLASS(p->then_branch.get(), "if stmt");
    NO_VAR_FUNC_CLASS(p->else_branch.get(), "if stmt");
  }
  if (auto p = ast_node->DynAs<ForStmt>()) {
    NO_VAR_FUNC_CLASS(p->body.get(), "for stmt");
    function_infos.back().scopes.push_back(ScopeInfo{.type = ScopeType::BLOCK});
  }
  if (auto p = ast_node->DynAs<WhileStmt>()) {
    NO_VAR_FUNC_CLASS(p->body.get(), "while stmt");
  }

  if (auto p = ast_node->DynAs<ClassStmt>()) {
    function_infos.back().scopes.push_back(ScopeInfo{.type = ScopeType::CLASS});
    all_classes[p->attr->name->lexeme] = std::make_shared<ClassInfo>();
    class_infos.push_back(all_classes[p->attr->name->lexeme]);
    if (p->superclass) {
      if (p->superclass->DynAs<VariableExpr>()->attr->name->lexeme == p->attr->name->lexeme) {
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Class can not inherit itself");
      }
      all_classes[p->attr->name->lexeme]->superclass =
          all_classes[p->superclass->DynAs<VariableExpr>()->attr->name->lexeme];
    }
    AddNamedValue(p->attr->name->lexeme);
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
    if (p->attr->name->lexeme == "super") {
      if (!class_infos.back()->superclass) {
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Can not use 'super' without base class");
      }
#ifdef UPSTREAM_STYLE_ERROR_MSG
      if (!ast_node->Parent()->DynAs<GetAttrExpr>() ||
          (ast_node->Parent()->As<GetAttrExpr>()->attr->attr_name->lexeme == "")) {
        // this check is only used to be compatible with up stream lox
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Lox does not allow 'super' in this context");
      }
#endif
    }
  }

  if (auto p = ast_node->DynAs<FunctionStmt>()) {
    if (function_infos.back().scopes.back().type == ScopeType::CLASS) {
      if (p->attr->name->lexeme == "init") {
        function_infos.emplace_back(FunctionInfo{FunctionType::INITIALIZER, p->attr->name->lexeme});
      } else {
        function_infos.emplace_back(FunctionInfo{FunctionType::METHOD, p->attr->name->lexeme});
      }
    } else {
      function_infos.emplace_back(FunctionInfo(FunctionType::FUNCTION, p->attr->name->lexeme));
    }
    AddNamedValue(p->attr->name->lexeme);
    if (p->comma_expr_params) {
      if (!p->comma_expr_params->DynAs<CommaExpr>()) {
        throw SemanticError("Semantic Error: " + p->attr->name->Dump() +
                            " Function parameter must be a comma separated list");
      }
      if (p->comma_expr_params->As<CommaExpr>()->elements.size() >= 255) {
        throw SemanticError("Lox Can't have more than 255 parameter.");
      } else {
        for (auto &param : p->comma_expr_params->As<CommaExpr>()->elements) {
          if (!param->DynAs<VariableExpr>()) {
            throw SemanticError("Semantic Error: " + p->attr->name->Dump() + " Function parameter must be variable");
          }
          AddNamedValue(param->As<VariableExpr>()->attr->name->lexeme);
        }
      }
    }
  }
  if (auto p = ast_node->DynAs<CallExpr>()) {
    if (p->comma_expr_args) {
      if (!p->comma_expr_args->DynAs<CommaExpr>()) {
        throw SemanticError("Semantic Error: Call arguments must be a comma separated list");
      }

      if (p->comma_expr_args->As<CommaExpr>()->elements.size() >= 255) {
        throw SemanticError("Lox Can't have more than 255 arguments.");
      }
    }
  }
  if (auto p = ast_node->DynAs<ReturnStmt>()) {
    if (function_infos.empty()) {
      throw SemanticError("Semantic Error: " + p->attr->src_token->Dump() + " Can't return from top-level code.");
    } else {
      bool is_in_function = false;
      for (auto scope_iter = function_infos.back().scopes.rbegin(); scope_iter != function_infos.back().scopes.rend();
           ++scope_iter) {
        if (scope_iter->type == ScopeType::FUNCTION) {
          is_in_function = true;
          break;
        }
        if (scope_iter->type == ScopeType::BLOCK) {
          continue;
        }
      }
      if (!is_in_function) {
        throw SemanticError("Semantic Error: " + p->attr->src_token->Dump() + " Can only return from function");
      }
    }

    if (function_infos.back().type == FunctionType::INITIALIZER) {
      if (p->value &&
          (!p->value->DynAs<VariableExpr>() || p->value->As<VariableExpr>()->attr->name->lexeme != "this")) {
        throw SemanticError("Semantic Error: " + p->attr->src_token->Dump() +
                            " Can't return value other than `this` from initializer.");
      }
    }
  }
  if (auto p = ast_node->DynAs<BlockStmt>()) {
    function_infos.back().scopes.push_back(ScopeInfo{.type = ScopeType::BLOCK});
  }

  // todo: check init must return nothing

  return Pass::IsModified::NO;
}
void SemanticCheck::AddNamedValue(const std::string &name) {
  if (function_infos.back().scopes.back().locals.contains(name) &&
      function_infos.back().scopes.back().type != ScopeType::GLOBAL) {
    throw SemanticError(std::string("Semantic Error:  Variable ") + name + " already declared.");
  }
  function_infos.back().scopes.back().locals.insert(name);
}
Pass::IsModified SemanticCheck::PostNode(ASTNode *ast_node) {
  if (auto p = ast_node->DynAs<WhileStmt>() || ast_node->DynAs<ForStmt>()) {
    loop_infos.pop_back();
  }
  if (auto p = ast_node->DynAs<FunctionStmt>()) {
    function_infos.pop_back();
  }
  if (auto p = ast_node->DynAs<ClassStmt>()) {
    class_infos.pop_back();
    function_infos.back().scopes.pop_back();
  }
  if (auto p = ast_node->DynAs<BlockStmt>()) {
    function_infos.back().scopes.pop_back();
  }
  if (auto p = ast_node->DynAs<ForStmt>()) {
    function_infos.back().scopes.pop_back();
  }

  return Pass::IsModified::NO;
}
} // namespace lox
