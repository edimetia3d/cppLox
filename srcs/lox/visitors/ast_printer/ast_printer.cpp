//
// License: MIT
//
#include "lox/visitors/ast_printer/ast_printer.h"

#include <map>

namespace lox {

object::LoxObject lox::ExprPrinter::Visit(BinaryState* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("(") + left_expr + op + right_expr + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(LiteralState* state) { return object::LoxObject(state->value.lexeme_); }
object::LoxObject lox::ExprPrinter::Visit(GroupingState* state) {
  return object::LoxObject(std::string("(") + Print(state->expression) + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(UnaryState* state) {
  return object::LoxObject(std::string("(") + state->op.lexeme_ + Print(state->right) + std::string(")"));
}
object::LoxObject ExprPrinter::Visit(VariableState* state) { return object::LoxObject(state->name.lexeme_); }
object::LoxObject ExprPrinter::Visit(AssignState* state) {
  return object::LoxObject(std::string("(") + state->name.lexeme_ + " = " + Print(state->value) + std::string(")"));
}

object::LoxObject StmtPrinter::Visit(PrintStmtState* state) {
  return object::LoxObject(std::string("print ") + expr_printer_.Print(state->expression) + ";");
}
object::LoxObject StmtPrinter::Visit(ExprStmtState* state) {
  return object::LoxObject(expr_printer_.Print(state->expression) + ";");
}
object::LoxObject StmtPrinter::Visit(VarDeclStmtState* state) {
  std::string init = "(NoInit)";
  if (state->initializer.IsValid()) {
    init = " = " + expr_printer_.Print(state->initializer);
  }
  return object::LoxObject(std::string("var ") + state->name.lexeme_ + init + ";");
}
namespace {
struct Level {
  struct V {
    int v = -1;
  };
  int Value() { return nest_level[current_printer].v; }
  Level(void* printer) : current_printer(printer) { nest_level[current_printer].v += 1; }
  ~Level() { nest_level[current_printer].v -= 1; }
  void* current_printer;
  static std::map<void*, V> nest_level;
};
std::map<void*, Level::V> Level::nest_level;
}  // namespace

object::LoxObject StmtPrinter::Visit(BlockStmtState* state) {
  Level level(this);
  std::string tab_base = "  ";
  std::string tab = "";
  for (int i = 0; i < level.Value(); ++i) {
    tab += tab_base;
  }
  std::string str = tab + "{\n";
  for (auto& stmt : state->statements) {
    str += (tab + Print(stmt));
    str += (tab + "\n");
  }
  str += (tab + "}");
  return object::LoxObject(str);
}
}  // namespace lox
