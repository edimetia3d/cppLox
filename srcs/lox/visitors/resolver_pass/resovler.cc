//
// LICENSE: MIT
//

#include "lox/visitors/resolver_pass/resovler.h"

#include "lox/error.h"

namespace lox {
static auto RETNULL = object::LoxObject::VoidObject();
object::LoxObject Resovler::Visit(BlockStmtState *state) {
  BeginScope();
  for (auto &stmt : state->statements) {
    Resolve(stmt);
  }
  EndScope();
  return RETNULL;
}
void Resovler::Declare(Token token) { scopes.back()[token.lexeme_] = false; }
object::LoxObject Resovler::Visit(VarDeclStmtState *state) {
  Declare(state->name);
  if (state->initializer.IsValid()) {
    Resolve(state->initializer);
  }
  Define(state->name);
  return RETNULL;
}
void Resovler::Define(Token token) { scopes.back()[token.lexeme_] = true; }
object::LoxObject Resovler::Visit(VariableState *state) {
  if (scopes.back().contains(state->name.lexeme_) and scopes.back()[state->name.lexeme_] == false) {
    throw ResolveError(Error(state->name, "Can't read local variable in its own initializer."));
  }

  ResolveLocal(state, state->name);
  return RETNULL;
}
void Resovler::ResolveLocal(ExprState *state, Token name) {
  for (int i = scopes.size() - 1; i >= 0; i--) {
    if (scopes[i].contains(name.lexeme_)) {
      map_->Set(state, scopes.size() - 1 - i);
      return;
    }
  }
}

object::LoxObject Resovler::Visit(AssignState *state) {
  Resolve(state->value);
  ResolveLocal(state, state->name);
  return RETNULL;
}
object::LoxObject Resovler::Visit(FunctionStmtState *state) {
  Declare(state->name);
  Define(state->name);

  ResolveFunction(state);
  return RETNULL;
}
void Resovler::ResolveFunction(FunctionStmtState *state) {
  BeginScope();
  for (Token param : state->params) {
    Declare(param);
    Define(param);
  }
  for (auto &stmt : state->body) {
    Resolve(stmt);
  }
  EndScope();
}
object::LoxObject Resovler::Visit(LogicalState *state) {
  Resolve(state->left);
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(BinaryState *state) {
  Resolve(state->left);
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(GroupingState *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(LiteralState *state) { return RETNULL; }
object::LoxObject Resovler::Visit(UnaryState *state) {
  Resolve(state->right);
  return RETNULL;
}
object::LoxObject Resovler::Visit(CallState *state) {
  for (auto &expr : state->arguments) {
    Resolve(expr);
  }
  Resolve(state->callee);
  return RETNULL;
}
object::LoxObject Resovler::Visit(PrintStmtState *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(ReturnStmtState *state) {
  Resolve(state->value);
  return RETNULL;
}
object::LoxObject Resovler::Visit(WhileStmtState *state) {
  Resolve(state->condition);
  Resolve(state->body);
  return RETNULL;
}
object::LoxObject Resovler::Visit(BreakStmtState *state) { return RETNULL; }
object::LoxObject Resovler::Visit(ExprStmtState *state) {
  Resolve(state->expression);
  return RETNULL;
}
object::LoxObject Resovler::Visit(IfStmtState *state) {
  Resolve(state->condition);
  Resolve(state->thenBranch);
  if (state->elseBranch.IsValid()) {
    Resolve(state->elseBranch);
  }

  return RETNULL;
}
object::LoxObject Resovler::Visit(ClassStmtState *state) {
  Declare(state->name);
  Define(state->name);
  return RETNULL;
}
object::LoxObject Resovler::Visit(GetAttrState *state) {
  Resolve(state->src_object);
  // this may be a recursive call, only top level name need be resovled, all attr will be "resolved" by GetAttr
  // at runtime.
  return RETNULL;
}
}  // namespace lox
