//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/passes/env_resolve_pass/env_reslove_pass.h"

#include "lox/backend/tree_walker/error.h"

namespace lox {

void EnvResovlePass::Declare(Token token) { scopes.back()[token->lexeme] = false; }

void EnvResovlePass::Define(Token token) { scopes.back()[token->lexeme] = true; }

void EnvResovlePass::ResolveName(ExprBase *state, Token name) {
  for (int i = scopes.size() - 1; i >= 0; i--) {
    if (scopes[i].contains(name->lexeme)) {
      map_->Set(state, scopes.size() - 1 - i);
      return;
    }
  }
}

void EnvResovlePass::PreNode(AstNode *ast_node, std::shared_ptr<AstNode> *replace_node) {
  if (MatchAnyType<BlockStmt>(ast_node)) {
    BeginScope(ScopeType::BLOCK);
    return;
  }
  if (auto p = CastTo<VarDeclStmt>(ast_node)) {
    Declare(p->name());
    return;
  }
  if (auto p = CastTo<VariableExpr>(ast_node)) {
    if (p->name()->type == TokenType::THIS) {
      if (!IsInClassScope() || current_function_type == FunctionType::NONE) {
        throw(ResolveError(p->name(), "Cannot read 'this' out of method"));
      }
    }
    if (scopes.back().contains(p->name()->lexeme) and scopes.back()[p->name()->lexeme] == false) {
      throw(ResolveError(p->name(), "Can't read local variable in its own initializer."));
    }

    ResolveName(p, p->name());
    return;
  }
  if (auto p = CastTo<FunctionStmt>(ast_node)) {
    Define(p->name());
    auto fn_type = FunctionType::FUNCTION;
    if (current_scope_type == ScopeType::CLASS) {
      // methods are attributes too, they are not name during resolve, and created at runtime.
      fn_type = FunctionType::METHOD;
      if (p->name()->lexeme == "init") {
        fn_type = FunctionType::INITIALIZER;
      }
    }

    previous_function_type.push_back(current_function_type);
    current_function_type = fn_type;
    BeginScope(ScopeType::FUNCTION);
    for (Token param : p->params()) {
      Define(param);
    }
    return;
  }
  if (auto p = CastTo<ReturnStmt>(ast_node)) {
    if (current_function_type == FunctionType::NONE) {
      throw(ResolveError(p->keyword(), "Cannot return at here"));
    }
    if (current_function_type == FunctionType::INITIALIZER) {
      if (IsValid(p->value())) {
        auto p2 = p->value()->DownCast<VariableExpr>();
        if (p2 == nullptr || p2->name()->lexeme != "this") {
          throw(ResolveError(p->keyword(), "INITIALIZER must return 'this'"));
        }
      }
    }
    return;
  }
  if (auto p = CastTo<ClassStmt>(ast_node)) {
    BeginScope(ScopeType::CLASS);
    auto token_this = MakeToken(TokenType::THIS, "this", p->name()->line);
    Define(token_this);
    Declare(p->name());
    return;
  }
}
void EnvResovlePass::PostNode(AstNode *ast_node, std::shared_ptr<AstNode> *replace_node) {
  if (MatchAnyType<BlockStmt>(ast_node)) {
    EndScope();
    return;
  }
  if (auto p = CastTo<VarDeclStmt>(ast_node)) {
    Define(p->name());
    return;
  }
  if (auto p = CastTo<AssignExpr>(ast_node)) {
    ResolveName(p, p->name());
    return;
  }
  if (auto p = CastTo<FunctionStmt>(ast_node)) {
    EndScope();
    current_function_type = previous_function_type.back();
    previous_function_type.pop_back();
    return;
  }
  if (auto p = CastTo<ClassStmt>(ast_node)) {
    Define(p->name());
    EndScope();
    return;
  }
}
}  // namespace lox
