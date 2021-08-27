//
// License: MIT
//
#include "lox/backend/tree_walker/visitors/ast_printer/ast_printer.h"

#include <map>

namespace lox {

void AstPrinter::Visit(LogicalExpr* state) {
  std::string left_expr = Print(state->left());
  std::string op = state->op()->lexeme;
  std::string right_expr = Print(state->right());
  VisitorReturn(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}

void lox::AstPrinter::Visit(BinaryExpr* state) {
  std::string left_expr = Print(state->left());
  std::string op = state->op()->lexeme;
  std::string right_expr = Print(state->right());
  VisitorReturn(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}
void AstPrinter::Visit(LiteralExpr* state) { VisitorReturn(state->value()->lexeme); }
void lox::AstPrinter::Visit(GroupingExpr* state) {
  VisitorReturn(std::string("(") + Print(state->expression()) + std::string(")"));
}
void AstPrinter::Visit(UnaryExpr* state) {
  VisitorReturn(std::string("(") + state->op()->lexeme + Print(state->right()) + std::string(")"));
}
void AstPrinter::Visit(VariableExpr* state) { VisitorReturn(state->name()->lexeme); }
void AstPrinter::Visit(AssignExpr* state) {
  VisitorReturn(std::string("(") + state->name()->lexeme + " = " + Print(state->value()) + std::string(")"));
}
void AstPrinter::Visit(CallExpr* state) {
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
  VisitorReturn(ret + ")");
}

void AstPrinter::Visit(PrintStmt* state) { VisitorReturn(std::string("print ") + Print(state->expression()) + ";"); }
void AstPrinter::Visit(ExprStmt* state) { VisitorReturn(Print(state->expression()) + ";"); }
void AstPrinter::Visit(VarDeclStmt* state) {
  std::string init = "(NoInit)";
  if (IsValid(state->initializer())) {
    init = " = " + Print(state->initializer());
  }
  VisitorReturn(std::string("var ") + state->name()->lexeme + init + ";");
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

void AstPrinter::Visit(BlockStmt* state) {
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
  VisitorReturn(str);
}
void AstPrinter::Visit(IfStmt* state) {
  std::string ret = "if ( " + Print(state->condition()) + " )\n";
  ret += "{\n" + Print(state->thenBranch()) + "}\n";
  if (IsValid(state->elseBranch())) {
    ret += "{\n" + Print(state->elseBranch()) + "}\n";
  }
  VisitorReturn(ret);
}
void AstPrinter::Visit(WhileStmt* state) {
  std::string ret = "while ( " + Print(state->condition()) + " )\n";
  ret += "{\n" + Print(state->body()) + "}\n";
  VisitorReturn(ret);
}
void AstPrinter::Visit(BreakStmt* state) { VisitorReturn(state->src_token()->lexeme); }
void AstPrinter::Visit(FunctionStmt* state) {
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
  VisitorReturn(ret);
}
void AstPrinter::Visit(ReturnStmt* state) {
  std::string ret = "return";
  if (IsValid(state->value())) {
    ret += Print(state->value());
  }
  VisitorReturn(ret + ";");
}
void AstPrinter::Visit(ClassStmt* state) {
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
  VisitorReturn(ret);
}
void AstPrinter::Visit(GetAttrExpr* state) {
  std::string ret = Print(state->src_object()) + "." + state->attr_name()->lexeme;
  VisitorReturn(ret);
}
void AstPrinter::Visit(SetAttrExpr* state) {
  std::string ret = Print(state->src_object()) + "." + state->attr_name()->lexeme;
  ret += " @= ";
  ret += Print(state->value());
  VisitorReturn(ret);
}

}  // namespace lox
