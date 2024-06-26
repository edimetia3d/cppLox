//
// License: MIT
//
#include "ast_printer.h"

namespace lox {

void AstPrinter::Visit(LogicalExpr *node) {
  std::string left_expr = Print(node->left.get());
  std::string op = node->attr->op->lexeme;
  std::string right_expr = Print(node->right.get());
  VisitorReturn(left_expr + " " + op + " " + right_expr);
}

void lox::AstPrinter::Visit(BinaryExpr *node) {
  std::string left_expr = Print(node->left.get());
  std::string op = node->attr->op->lexeme;
  std::string right_expr = Print(node->right.get());
  VisitorReturn(left_expr + " " + op + " " + right_expr);
}
void AstPrinter::Visit(LiteralExpr *node) { VisitorReturn(node->attr->value->lexeme); }
void lox::AstPrinter::Visit(GroupingExpr *node) {
  VisitorReturn(std::string("(") + Print(node->expression.get()) + ")");
}
void AstPrinter::Visit(UnaryExpr *node) { VisitorReturn(node->attr->op->lexeme.Str() + Print(node->right.get())); }
void AstPrinter::Visit(VariableExpr *node) { VisitorReturn(node->attr->name->lexeme); }
void AstPrinter::Visit(AssignExpr *node) {
  VisitorReturn(node->attr->name->lexeme.Str() + " = " + Print(node->value.get()));
}
void AstPrinter::Visit(CallExpr *node) {
  std::string ret = "";
  ret = ret + Print(node->callee.get()) + "(";
  ret += Print(node->comma_expr_args.get());
  VisitorReturn(ret + ")");
}

void AstPrinter::Visit(PrintStmt *node) {
  VisitorReturn(Indentation() + std::string("print ") + Print(node->expression.get()) + ";\n");
}
void AstPrinter::Visit(ExprStmt *node) { VisitorReturn(Indentation() + Print(node->expression.get()) + ";\n"); }
void AstPrinter::Visit(VarDeclStmt *node) {
  std::string init = "";
  if (node->initializer) {
    init = " = " + Print(node->initializer.get());
  }
  VisitorReturn(Indentation() + std::string("var ") + node->attr->name->lexeme.Str() + init + ";\n");
}
void AstPrinter::Visit(BlockStmt *node) {
  auto indentation = Indentation();
  std::string ret = indentation + "{\n";
  SemanticLevelGuard guard(this);
  for (auto &stmt : node->statements) {
    ret += Print(stmt.get());
  }
  ret += (indentation + "}\n");
  VisitorReturn(ret);
}
void AstPrinter::Visit(IfStmt *node) {
  auto indentation = Indentation();
  std::string ret = indentation + "if (" + Print(node->condition.get()) + ")";
  PossibleBlockPrint(node->then_branch.get(), ret);
  if (node->else_branch) {
    ret += (indentation + "else ");
    PossibleBlockPrint(node->else_branch.get(), ret);
  }
  VisitorReturn(ret);
}
void AstPrinter::Visit(WhileStmt *node) {
  std::string ret = Indentation() + "while (" + Print(node->condition.get()) + ")\n";
  PossibleBlockPrint(node->body.get(), ret);
  VisitorReturn(ret);
}
void AstPrinter::Visit(ForStmt *node) {
  std::string ret = Indentation() + "for (";
  if (node->initializer) {
    if (node->initializer->DynAs<VarDeclStmt>()) {
      auto var_decl = node->initializer->As<VarDeclStmt>();
      std::string init = "";
      if (var_decl->initializer) {
        init = std::string(" = ") + Print(var_decl->initializer.get());
      }
      ret += "var " + var_decl->attr->name->lexeme.Str() + init + "; ";
    } else {
      ret += Print(node->initializer.get()) + "; ";
    }
  }
  if (node->condition) {
    ret += Print(node->condition.get()) + "; ";
  }
  if (node->increment) {
    ret += Print(node->increment.get()) + ")";
  }
  PossibleBlockPrint(node->body.get(), ret);
  VisitorReturn(ret);
}

std::string &AstPrinter::PossibleBlockPrint(ASTNode *node, std::string &ret) {
  if (node->DynAs<BlockStmt>()) {
    ret += Print(node);
  } else {
    ret += "\n";
    SemanticLevelGuard guard(this);
    ret += Print(node);
  }
  return ret;
}
void AstPrinter::Visit(BreakStmt *node) { VisitorReturn(node->attr->src_token->lexeme); }
void AstPrinter::Visit(FunctionStmt *node) {
  auto indentation = Indentation();
  std::string ret = indentation + "fun " + node->attr->name->lexeme.Str() + "(";
  if (node->comma_expr_params) {
    ret += Print(node->comma_expr_params.get());
  }
  ret += ") {\n";
  SemanticLevelGuard guard(this);
  for (auto &stmt : node->body) {
    ret += Print(stmt.get());
  }
  ret += (indentation + "}\n");
  VisitorReturn(ret);
}
void AstPrinter::Visit(ReturnStmt *node) {
  std::string ret = Indentation() + "return ";
  if (node->value) {
    ret += Print(node->value.get());
  }
  VisitorReturn(ret + ";\n");
}
void AstPrinter::Visit(ClassStmt *node) {
  auto indentation = Indentation();
  std::string ret = indentation + "class " + node->attr->name->lexeme.Str();
  if (node->superclass.get()) {
    ret += " < ";
    ret += Print(node->superclass.get());
  }
  ret += "{\n";
  SemanticLevelGuard guard(this);
  for (auto &method : node->methods) {
    ret += Print(method.get());
  }
  ret += (indentation + "}\n");
  VisitorReturn(ret);
}
void AstPrinter::Visit(GetAttrExpr *node) {
  std::string ret = Print(node->src_object.get()) + "." + node->attr->attr_name->lexeme.Str();
  VisitorReturn(ret);
}
void AstPrinter::Visit(SetAttrExpr *node) {
  std::string ret = Print(node->src_object.get()) + "." + node->attr->attr_name->lexeme.Str();
  ret += " = ";
  ret += Print(node->value.get());
  VisitorReturn(ret);
}
void AstPrinter::Visit(CommaExpr *node) {
  std::string ret = "";
  int i = 0;
  for (auto &expr : node->elements) {
    if (i > 0) {
      ret += ", ";
    }
    ret += Print(expr.get());
    ++i;
  }
  VisitorReturn(ret);
}

void AstPrinter::Visit(ListExpr *node) {
  std::string ret = "[";
  if (node->comma_expr) {
    ret += Print(node->comma_expr.get());
  }
  ret += "]";
  VisitorReturn(ret);
}
void AstPrinter::Visit(GetItemExpr *node) {
  std::string ret = Print(node->src_object.get()) + "[" + Print(node->index.get()) + "]";
  VisitorReturn(ret);
}
void AstPrinter::Visit(TensorExpr *node) {
  std::string ret = "Tensor(";
  ret += Print(node->dtype.get()) + ", ";
  ret += Print(node->shape.get()) + ",";
  ret += Print(node->data.get());
  ret += ")";
  VisitorReturn(ret);
}

} // namespace lox
