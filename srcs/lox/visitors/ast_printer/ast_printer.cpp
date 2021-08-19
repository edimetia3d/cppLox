//
// License: MIT
//
#include "lox/visitors/ast_printer/ast_printer.h"

#include <map>

namespace lox {

object::LoxObject AstPrinter::Visit(LogicalExpr* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}

object::LoxObject lox::AstPrinter::Visit(BinaryExpr* state) {
  std::string left_expr = Print(state->left);
  std::string op = state->op.lexeme_;
  std::string right_expr = Print(state->right);
  return object::LoxObject(std::string("( ") + left_expr + " " + op + " " + right_expr + std::string(" )"));
}
object::LoxObject AstPrinter::Visit(LiteralExpr* state) { return object::LoxObject(state->value.lexeme_); }
object::LoxObject lox::AstPrinter::Visit(GroupingExpr* state) {
  return object::LoxObject(std::string("(") + Print(state->expression) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(UnaryExpr* state) {
  return object::LoxObject(std::string("(") + state->op.lexeme_ + Print(state->right) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(VariableExpr* state) { return object::LoxObject(state->name.lexeme_); }
object::LoxObject AstPrinter::Visit(AssignExpr* state) {
  return object::LoxObject(std::string("(") + state->name.lexeme_ + " = " + Print(state->value) + std::string(")"));
}
object::LoxObject AstPrinter::Visit(CallExpr* state) {
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

object::LoxObject AstPrinter::Visit(PrintStmt* state) {
  return object::LoxObject(std::string("print ") + Print(state->expression) + ";");
}
object::LoxObject AstPrinter::Visit(ExprStmt* state) { return object::LoxObject(Print(state->expression) + ";"); }
object::LoxObject AstPrinter::Visit(VarDeclStmt* state) {
  std::string init = "(NoInit)";
  if (IsValid(state->initializer)) {
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

object::LoxObject AstPrinter::Visit(BlockStmt* state) {
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
object::LoxObject AstPrinter::Visit(IfStmt* state) {
  std::string ret = "if ( " + Print(state->condition) + " )\n";
  ret += "{\n" + Print(state->thenBranch) + "}\n";
  if (IsValid(state->elseBranch)) {
    ret += "{\n" + Print(state->elseBranch) + "}\n";
  }
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(WhileStmt* state) {
  std::string ret = "while ( " + Print(state->condition) + " )\n";
  ret += "{\n" + Print(state->body) + "}\n";
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(BreakStmt* state) { return object::LoxObject(state->src_token.lexeme_); }
object::LoxObject AstPrinter::Visit(FunctionStmt* state) {
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
object::LoxObject AstPrinter::Visit(ReturnStmt* state) {
  std::string ret = "return";
  if (IsValid(state->value)) {
    ret += Print(state->value);
  }
  return object::LoxObject(ret + ";");
}
object::LoxObject AstPrinter::Visit(ClassStmt* state) {
  std::string ret = "class ";
  ret += state->name.lexeme_ + "{\n";
  for (auto& method : state->methods) {
    ret += Print(method);
  }
  ret += "}";
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(GetAttrExpr* state) {
  std::string ret = Print(state->src_object) + "." + state->attr_name.lexeme_;
  return object::LoxObject(ret);
}
object::LoxObject AstPrinter::Visit(SetAttrExpr* state) {
  std::string ret = Print(state->src_object) + "." + state->attr_name.lexeme_;
  ret += " @= ";
  ret += Print(state->value);
  return object::LoxObject(ret);
}

}  // namespace lox
