Expr{
    "Binary": "Expr left, Token op, Expr right",
    "Grouping": "Expr expression",
    "Literal": "Token value",
    "Unary": "Token op, Expr right",
    "Variable" : "Token name",
    "Assign"   : "Token name, Expr value",
}
Stmt{
    "ExprStmt"       : "Expr expression",
    "PrintStmt"      : "Expr expression",
    "VarDeclStmt"    : "Token name, Expr initializer",
    "BlockStmt"      : "std::vector<Stmt> statements",
}
