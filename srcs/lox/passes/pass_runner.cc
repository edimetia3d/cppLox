//
// LICENSE: MIT
//

#include "pass_runner.h"
#define RUNPASS_AND_UPDATE(KEY_NAME)              \
  {                                               \
    auto old_value = (KEY_NAME)();                \
    auto new_value = RunPass(old_value);          \
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
        node = new_value;                     \
      };                                      \
    }                                         \
    if ((KEY_NAME)() != cpy) {                \
      (KEY_NAME)(cpy);                        \
    }                                         \
  }
namespace lox {
void PassRunner::Visit(BlockStmt *state) { RUNPASS_ON_VEC_AND_UPDATE(state->statements) }
void PassRunner::Visit(VarDeclStmt *state) {
  if (IsValid(state->initializer())) {
    RUNPASS_AND_UPDATE(state->initializer);
  }
}
void PassRunner::Visit(VariableExpr *state) {}

void PassRunner::Visit(AssignExpr *state) { RUNPASS_AND_UPDATE(state->value); }
void PassRunner::Visit(FunctionStmt *state) { RUNPASS_ON_VEC_AND_UPDATE(state->body); }
void PassRunner::Visit(LogicalExpr *state) {
  RUNPASS_AND_UPDATE(state->left);
  RUNPASS_AND_UPDATE(state->right);
}
void PassRunner::Visit(BinaryExpr *state) {
  RUNPASS_AND_UPDATE(state->left);
  RUNPASS_AND_UPDATE(state->right);
}
void PassRunner::Visit(GroupingExpr *state) { RUNPASS_AND_UPDATE(state->expression); }
void PassRunner::Visit(LiteralExpr *state) {}
void PassRunner::Visit(UnaryExpr *state) { RUNPASS_AND_UPDATE(state->right); }
void PassRunner::Visit(CallExpr *state) {
  RUNPASS_AND_UPDATE(state->callee);
  RUNPASS_ON_VEC_AND_UPDATE(state->arguments)
}
void PassRunner::Visit(PrintStmt *state) { RUNPASS_AND_UPDATE(state->expression); }
void PassRunner::Visit(ReturnStmt *state) {
  if (IsValid(state->value())) {
    RUNPASS_AND_UPDATE(state->value);
  }
}
void PassRunner::Visit(WhileStmt *state) {
  RUNPASS_AND_UPDATE(state->condition);
  RUNPASS_AND_UPDATE(state->body);
}
void PassRunner::Visit(BreakStmt *state) {}
void PassRunner::Visit(ExprStmt *state) { RUNPASS_AND_UPDATE(state->expression); }
void PassRunner::Visit(IfStmt *state) {
  RUNPASS_AND_UPDATE(state->condition);
  RUNPASS_AND_UPDATE(state->thenBranch);
  if (IsValid(state->elseBranch())) {
    RUNPASS_AND_UPDATE(state->elseBranch);
  }
}
void PassRunner::Visit(ClassStmt *state) {
  if (IsValid(state->superclass())) {
    RUNPASS_AND_UPDATE(state->superclass);
  }
  RUNPASS_ON_VEC_AND_UPDATE(state->methods)
}
void PassRunner::Visit(GetAttrExpr *state) { RUNPASS_AND_UPDATE(state->src_object); }
void PassRunner::Visit(SetAttrExpr *state) {
  RUNPASS_AND_UPDATE(state->src_object);
  RUNPASS_AND_UPDATE(state->value);
}
}  // namespace lox
