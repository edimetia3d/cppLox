//
// License: MIT
//
#include "lox/visitors/ast_printer/ast_printer.h"

#include <map>

namespace lox {

object::LoxObject AstPrinter::Visit(LogicalState* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}

object::LoxObject lox::AstPrinter::Visit(BinaryState* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}
object::LoxObject AstPrinter::Visit(LiteralState* state) { return object::LoxObject(state->value.lexeme_); }
object::LoxObject lox::AstPrinter::Visit(GroupingState* state) {
  return object::LoxObject(std::string("(") + Print(state->expression) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(UnaryState* state) {
  return object::LoxObject(std::string("(") + state->op.lexeme_ + Print(state->right) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(VariableState* state) { return object::LoxObject(state->name.lexeme_); }
object::LoxObject AstPrinter::Visit(AssignState* state) {
  return object::LoxObject(std::string("(") + state->name.lexeme_ + " = " + Print(state->value) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(CallState* state) {
  std::string ret = "";
  ret = ret + Print(state->callee) + "(";
  int i = 0;
  for (auto& arg : state->arguments) {
    if (i > 0) {
      ret += ",";
    }
    ret += Print(arg);
    ++i;
  }
  return object::LoxObject(ret + ")");
}

object::LoxObject AstPrinter::Visit(PrintStmtState* state) {
  return object::LoxObject(std::string("print ") + Print(state->expression) + ";");
}
object::LoxObject AstPrinter::Visit(ExprStmtState* state) { return object::LoxObject(Print(state->expression) + ";"); }
object::LoxObject AstPrinter::Visit(VarDeclStmtState* state) {
  std::string init = "(NoInit)";
  if (state->initializer.IsValid()) {
    init = " = " + Print(state->initializer);
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

object::LoxObject AstPrinter::Visit(BlockStmtState* state) {
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
object::LoxObject AstPrinter::Visit(IfStmtState* state) {
  std::string ret = "if ( " + Print(state->condition) + " )\n";
  ret += "{\n" + Print(state->thenBranch) + "}\n";
  if (state->elseBranch.IsValid()) {
    ret += "{\n" + Print(state->elseBranch) + "}\n";
  }
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(WhileStmtState* state) {
  std::string ret = "while ( " + Print(state->condition) + " )\n";
  ret += "{\n" + Print(state->body) + "}\n";
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(BreakStmtState* state) { return object::LoxObject(state->src_token.lexeme_); }
object::LoxObject AstPrinter::Visit(FunctionStmtState* state) {
  std::string ret = "fun ";
  ret += state->name.lexeme_ + " (";
  int i = 0;
  for (auto& param : state->params) {
    if (i > 0) {
      ret += ",";
    }
    ret += param.lexeme_;
    ++i;
  }
  ret += "){\n";
  for (auto& stmt : state->body) {
    ret += Print(stmt);
    ret += "\n";
  }
  ret += "}";
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(ReturnStmtState* state) {
  std::string ret = "return";
  if (state->value.IsValid()) {
    ret += Print(state->value);
  }
  return object::LoxObject(ret + ";");
}
object::LoxObject AstPrinter::Visit(ClassStmtState* state) {
  std::string ret = "class ";
  ret += state->name.lexeme_ + "{}";
  return object::LoxObject(ret);
}

}  // namespace lox
