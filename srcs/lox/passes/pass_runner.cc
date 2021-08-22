//
// LICENSE: MIT
//

#include "lox/passes/pass_runner.h"
#define RUNPASS_AND_UPDATE(KEY_NAME)            \
  {                                             \
    auto new_value = RunPass((KEY_NAME));       \
    new_value->SetParent((KEY_NAME)->Parent()); \
    (KEY_NAME) = new_value;                     \
  }
namespace lox {
static auto RETNULL = object::VoidObject();
object::LoxObject PassRunner::Visit(BlockStmt *state) {
  for (auto &stmt : state->statements) {
    RUNPASS_AND_UPDATE(stmt);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(VarDeclStmt *state) {
  if (IsValid(state->initializer)) {
    RUNPASS_AND_UPDATE(state->initializer);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(VariableExpr *state) { return RETNULL; }

object::LoxObject PassRunner::Visit(AssignExpr *state) {
  RUNPASS_AND_UPDATE(state->value);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(FunctionStmt *state) {
  for (auto &body_stmt : state->body) {
    RUNPASS_AND_UPDATE(body_stmt);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(LogicalExpr *state) {
  RUNPASS_AND_UPDATE(state->left);
  RUNPASS_AND_UPDATE(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(BinaryExpr *state) {
  RUNPASS_AND_UPDATE(state->left);
  RUNPASS_AND_UPDATE(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(GroupingExpr *state) {
  RUNPASS_AND_UPDATE(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(LiteralExpr *state) { return RETNULL; }
object::LoxObject PassRunner::Visit(UnaryExpr *state) {
  RUNPASS_AND_UPDATE(state->right);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(CallExpr *state) {
  RUNPASS_AND_UPDATE(state->callee);
  for (auto &expr : state->arguments) {
    RUNPASS_AND_UPDATE(expr);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(PrintStmt *state) {
  RUNPASS_AND_UPDATE(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ReturnStmt *state) {
  if (IsValid(state->value)) {
    RUNPASS_AND_UPDATE(state->value);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(WhileStmt *state) {
  RUNPASS_AND_UPDATE(state->condition);
  RUNPASS_AND_UPDATE(state->body);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(BreakStmt *state) { return RETNULL; }
object::LoxObject PassRunner::Visit(ExprStmt *state) {
  RUNPASS_AND_UPDATE(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(IfStmt *state) {
  RUNPASS_AND_UPDATE(state->condition);
  RUNPASS_AND_UPDATE(state->thenBranch);
  if (IsValid(state->elseBranch)) {
    RUNPASS_AND_UPDATE(state->elseBranch);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ClassStmt *state) {
  if (IsValid(state->superclass)) {
    RUNPASS_AND_UPDATE(state->superclass);
  }
  for (auto &method : state->methods) {
    RUNPASS_AND_UPDATE(method);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(GetAttrExpr *state) {
  RUNPASS_AND_UPDATE(state->src_object);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(SetAttrExpr *state) {
  RUNPASS_AND_UPDATE(state->src_object);
  RUNPASS_AND_UPDATE(state->value);
  return RETNULL;
}
}  // namespace lox
