//
// LICENSE: MIT
//

#include "lox/passes/pass_runner.h"
#define RUNPASS_AND_UPDATE(KEY_NAME)              \
  {                                               \
    auto old_value = (KEY_NAME)();                \
    auto new_value = RunPass(old_value);          \
    new_value->SetParent((KEY_NAME)()->Parent()); \
    if (new_value != old_value) {                 \
      (KEY_NAME)(new_value);                      \
    }                                             \
  }

#define RUNPASS_ON_VEC_AND_UPDATE(KEY_NAME)   \
  {                                           \
    auto cpy = (KEY_NAME)();                  \
    for (auto &node : cpy) {                  \
      {                                       \
        auto new_value = RunPass(node);       \
        new_value->SetParent(node->Parent()); \
        node = new_value;                     \
      };                                      \
    }                                         \
    if ((KEY_NAME)() != cpy) {                \
      (KEY_NAME)(cpy);                        \
    }                                         \
  }
namespace lox {
static auto RETNULL = object::VoidObject();
object::LoxObject PassRunner::Visit(BlockStmt *state) {
  RUNPASS_ON_VEC_AND_UPDATE(state->statements)
  return RETNULL;
}
object::LoxObject PassRunner::Visit(VarDeclStmt *state) {
  if (IsValid(state->initializer())) {
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
  RUNPASS_ON_VEC_AND_UPDATE(state->body);
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
  RUNPASS_ON_VEC_AND_UPDATE(state->arguments)

  return RETNULL;
}
object::LoxObject PassRunner::Visit(PrintStmt *state) {
  RUNPASS_AND_UPDATE(state->expression);
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ReturnStmt *state) {
  if (IsValid(state->value())) {
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
  if (IsValid(state->elseBranch())) {
    RUNPASS_AND_UPDATE(state->elseBranch);
  }
  return RETNULL;
}
object::LoxObject PassRunner::Visit(ClassStmt *state) {
  if (IsValid(state->superclass())) {
    RUNPASS_AND_UPDATE(state->superclass);
  }
  RUNPASS_ON_VEC_AND_UPDATE(state->methods)
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
