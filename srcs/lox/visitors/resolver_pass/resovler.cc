//
// LICENSE: MIT
//

#include "lox/visitors/resolver_pass/resovler.h"

#include "lox/error.h"

namespace lox {
static auto RETNULL = object::VoidObject();
object::LoxObject Resovler::Visit(BlockStmt *state) {
  BeginScope();
  for (auto &stmt : state->statements) {
    Resolve(stmt);
  }
  EndScope();
  return RETNULL;
}
void Resovler::Declare(Token token) { scopes.back()[token.lexeme_] = false; }
object::LoxObject Resovler::Visit(VarDeclStmt *state) {
  Declare(state->name);
  if (IsValid(state->initializer)) {
    Resolve(state->initializer);
  }
  Define(state->name);
  return RETNULL;
}
void Resovler::Define(Token token) { scopes.back()[token.lexeme_] = true; }
object::LoxObject Resovler::Visit(VariableExpr *state) {
  if (state->name.type_ == TokenType::THIS) {
    if (current_function_type != FunctionType::METHOD && current_function_type != FunctionType::INITIALIZER) {
      throw ResolveError(Error(state->name, "Cannot read 'this' out of method"));
    }
  }
  if (scopes.back().contains(state->name.lexeme_) and scopes.back()[state->name.lexeme_] == false) {
    throw ResolveError(Error(state->name, "Can't read local variable in its own initializer."));
  }

  ResolveLocal(state, state->name);
  return RETNULL;
}
void Resovler::ResolveLocal(ExprBase *state, Token name) {
  for (int i = scopes.size() - 1; i >= 0; i--) {
    if (scopes[i].contains(name.lexeme_)) {
      map_->Set(state, scopes.size() - 1 - i);
      return;
    }
  }
}

object::LoxObject Resovler::Visit(AssignExpr *state) {
  Resolve(state->value);
  ResolveLocal(state, state->name);
  return RETNULL;
}
object::LoxObject Resovler::Visit(FunctionStmt *state) {
  Declare(state->name);
  Define(state->name);

  ResolveFunction(state, FunctionType::FUNCTION);
  return RETNULL;
}
void Resovler::ResolveFunction(FunctionStmt *state, FunctionType type) {
  auto previous_type = current_function_type;
  current_function_type = type;
  BeginScope();
  for (Token param : state->params) {
    Declare(param);
    Define(param);
  }
  for (auto &stmt : state->body) {
    Resolve(stmt);
  }
  EndScope();
  current_function_type = previous_type;
}
object::LoxObject Resovler::Visit(LogicalExpr *state) {
  Resolve(state->left);
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(BinaryExpr *state) {
  Resolve(state->left);
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(GroupingExpr *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(LiteralExpr *state) { return RETNULL; }
object::LoxObject Resovler::Visit(UnaryExpr *state) {
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(CallExpr *state) {
  for (auto &expr : state->arguments) {
    Resolve(expr);
  }
  Resolve(state->callee);
  return RETNULL;
}
object::LoxObject Resovler::Visit(PrintStmt *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(ReturnStmt *state) {
  if (current_function_type == FunctionType::NONE) {
    throw ResolveError(Error(state->keyword, "Cannot return at here"));
  }
  if (current_function_type == FunctionType::INITIALIZER) {
    if (IsValid(state->value)) {
      auto p = state->value->DownCast<VariableExpr>();
      if (p == nullptr || p->name.lexeme_ != "this") {
        throw ResolveError(Error(state->keyword, "INITIALIZER must return 'this'"));
      }
    }
  }
  if (IsValid(state->value)) {
    Resolve(state->value);
  }
  return RETNULL;
}
object::LoxObject Resovler::Visit(WhileStmt *state) {
  ++while_loop_level;
  Resolve(state->condition);
  Resolve(state->body);
  --while_loop_level;
  return RETNULL;
}
object::LoxObject Resovler::Visit(BreakStmt *state) {
  if (while_loop_level == 0) {
    throw ResolveError(Error(state->src_token, "Nothing to break here."));
  }
  return RETNULL;
}
object::LoxObject Resovler::Visit(ExprStmt *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(IfStmt *state) {
  Resolve(state->condition);
  Resolve(state->thenBranch);
  if (IsValid(state->elseBranch)) {
    Resolve(state->elseBranch);
  }

  return RETNULL;
}
object::LoxObject Resovler::Visit(ClassStmt *state) {
  BeginScope();
  auto token_this = Token(TokenType::THIS, "this", state->name.line_);
  Define(token_this);
  Declare(state->name);
  for (auto fn_decl : state->methods) {
    auto fn_type = FunctionType::METHOD;
    // methods are attributes too, they are not name during resolve, and created at runtime.
    auto fn_state = fn_decl->DownCast<FunctionStmt>();
    if (fn_state->name.lexeme_ == "init") {
      fn_type = FunctionType::INITIALIZER;
    }
    ResolveFunction(fn_state, fn_type);
  }
  Define(state->name);
  EndScope();
  return RETNULL;
}
object::LoxObject Resovler::Visit(GetAttrExpr *state) {
  Resolve(state->src_object);
  // attr are not name during resolve, and created at runtime. so no need to call Define here
  // this may be a recursive call, only top level name need be resovled, all attr will be "resolved" by GetAttr
  // at runtime.
  return RETNULL;
}
object::LoxObject Resovler::Visit(SetAttrExpr *state) {
  // attr are not name during resolve, and created at runtime. so no need to call Define here
  Resolve(state->src_object);
  Resolve(state->value);
  return RETNULL;
}
}  // namespace lox
