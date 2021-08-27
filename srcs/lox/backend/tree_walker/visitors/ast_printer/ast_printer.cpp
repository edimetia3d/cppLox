//
// License: MIT
//
#include "lox/backend/tree_walker/visitors/ast_printer/ast_printer.h"

#include <map>

namespace lox {

object::LoxObject AstPrinter::Visit(LogicalExpr* state) {
  std::string left_expr = Print(state->left());
  std::string op = state->op()->lexeme;
  std::string right_expr = Print(state->right());
  return object::MakeLoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}

object::LoxObject lox::AstPrinter::Visit(BinaryExpr* state) {
  std::string left_expr = Print(state->left());
  std::string op = state->op()->lexeme;
  std::string right_expr = Print(state->right());
  return object::MakeLoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}
object::LoxObject AstPrinter::Visit(LiteralExpr* state) { return object::MakeLoxObject(state->value()->lexeme); }
object::LoxObject lox::AstPrinter::Visit(GroupingExpr* state) {
  return object::MakeLoxObject(std::string("(") + Print(state->expression()) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(UnaryExpr* state) {
  return object::MakeLoxObject(std::string("(") + state->op()->lexeme + Print(state->right()) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(VariableExpr* state) { return object::MakeLoxObject(state->name()->lexeme); }
object::LoxObject AstPrinter::Visit(AssignExpr* state) {
  return object::MakeLoxObject(std::string("(") + state->name()->lexeme + " = " + Print(state->value()) +
                               std::string(")"));
}
object::LoxObject AstPrinter::Visit(CallExpr* state) {
  std::string ret = "";
  ret = ret + Print(state->callee()) + "(";
  int i = 0;
  for (auto& arg : state->arguments()) {
    if (i > 0) {
      ret += ",";
    }
    ret += Print(arg);
    ++i;
  }
  return object::MakeLoxObject(ret + ")");
}

object::LoxObject AstPrinter::Visit(PrintStmt* state) {
  return object::MakeLoxObject(std::string("print ") + Print(state->expression()) + ";");
}
object::LoxObject AstPrinter::Visit(ExprStmt* state) { return object::MakeLoxObject(Print(state->expression()) + ";"); }
object::LoxObject AstPrinter::Visit(VarDeclStmt* state) {
  std::string init = "(NoInit)";
  if (IsValid(state->initializer())) {
    init = " = " + Print(state->initializer());
  }
  return object::MakeLoxObject(std::string("var ") + state->name()->lexeme + init + ";");
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

object::LoxObject AstPrinter::Visit(BlockStmt* state) {
  Level level(this);
  std::string tab_base = "  ";
  std::string tab = "";
  for (int i = 0; i < level.Value(); ++i) {
    tab += tab_base;
  }
  std::string str = tab + "{\n";
  for (auto& stmt : state->statements()) {
    str += (tab + Print(stmt));
    str += (tab + "\n");
  }
  str += (tab + "}");
  return object::MakeLoxObject(str);
}
object::LoxObject AstPrinter::Visit(IfStmt* state) {
  std::string ret = "if ( " + Print(state->condition()) + " )\n";
  ret += "{\n" + Print(state->thenBranch()) + "}\n";
  if (IsValid(state->elseBranch())) {
    ret += "{\n" + Print(state->elseBranch()) + "}\n";
  }
  return object::MakeLoxObject(ret);
}
object::LoxObject AstPrinter::Visit(WhileStmt* state) {
  std::string ret = "while ( " + Print(state->condition()) + " )\n";
  ret += "{\n" + Print(state->body()) + "}\n";
  return object::MakeLoxObject(ret);
}
object::LoxObject AstPrinter::Visit(BreakStmt* state) { return object::MakeLoxObject(state->src_token()->lexeme); }
object::LoxObject AstPrinter::Visit(FunctionStmt* state) {
  std::string ret = "fun ";
  ret += state->name()->lexeme + " (";
  int i = 0;
  for (auto& param : state->params()) {
    if (i > 0) {
      ret += ",";
    }
    ret += param->lexeme;
    ++i;
  }
  ret += "){\n";
  for (auto& stmt : state->body()) {
    ret += Print(stmt);
    ret += "\n";
  }
  ret += "}";
  return object::MakeLoxObject(ret);
}
object::LoxObject AstPrinter::Visit(ReturnStmt* state) {
  std::string ret = "return";
  if (IsValid(state->value())) {
    ret += Print(state->value());
  }
  return object::MakeLoxObject(ret + ";");
}
object::LoxObject AstPrinter::Visit(ClassStmt* state) {
  std::string ret = "class ";
  ret += state->name()->lexeme;
  if (IsValid(state->superclass())) {
    ret += " < ";
    ret += Print(state->superclass());
  }
  ret += "{\n";
  for (auto& method : state->methods()) {
    ret += Print(method);
  }
  ret += "}";
  return object::MakeLoxObject(ret);
}
object::LoxObject AstPrinter::Visit(GetAttrExpr* state) {
  std::string ret = Print(state->src_object()) + "." + state->attr_name()->lexeme;
  return object::MakeLoxObject(ret);
}
object::LoxObject AstPrinter::Visit(SetAttrExpr* state) {
  std::string ret = Print(state->src_object()) + "." + state->attr_name()->lexeme;
  ret += " @= ";
  ret += Print(state->value());
  return object::MakeLoxObject(ret);
}

}  // namespace lox
