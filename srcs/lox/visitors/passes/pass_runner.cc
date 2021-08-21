//
// LICENSE: MIT
//

#include "lox/visitors/passes/pass_runner.h"

namespace lox {
static auto RETNULL = object::VoidObject();
object::LoxObject PassRunner::Visit(BlockStmt *state) {
  for (auto &stmt : state->statements) {
    RunPass(stmt);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(VarDeclStmt *state) {
  if (IsValid(state->initializer)) {
    RunPass(state->initializer);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(VariableExpr *state) { return RETNULL; }

object::LoxObject PassRunner::Visit(AssignExpr *state) {
  RunPass(state->value);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(FunctionStmt *state) {
  for (auto &body_stmt : state->body) {
    RunPass(body_stmt);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(LogicalExpr *state) {
  RunPass(state->left);
  RunPass(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(BinaryExpr *state) {
  RunPass(state->left);
  RunPass(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(GroupingExpr *state) {
  RunPass(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(LiteralExpr *state) { return RETNULL; }
object::LoxObject PassRunner::Visit(UnaryExpr *state) {
  RunPass(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(CallExpr *state) {
  RunPass(state->callee);
  for (auto &expr : state->arguments) {
    RunPass(expr);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(PrintStmt *state) {
  RunPass(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ReturnStmt *state) {
  if (IsValid(state->value)) {
    RunPass(state->value);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(WhileStmt *state) {
  RunPass(state->condition);
  RunPass(state->body);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(BreakStmt *state) { return RETNULL; }
object::LoxObject PassRunner::Visit(ExprStmt *state) {
  RunPass(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(IfStmt *state) {
  RunPass(state->condition);
  RunPass(state->thenBranch);
  if (IsValid(state->elseBranch)) {
    RunPass(state->elseBranch);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ClassStmt *state) {
  RunPass(state->superclass);
  for (auto &method : state->methods) {
    RunPass(method);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(GetAttrExpr *state) {
  RunPass(state->src_object);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(SetAttrExpr *state) {
  RunPass(state->src_object);
  RunPass(state->value);
  return RETNULL;
}
}  // namespace lox
