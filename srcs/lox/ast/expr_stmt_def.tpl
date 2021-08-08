Expr{
    "Assign"   : "Token name, Expr value",
    "Logical"  : "Expr left, Token op, Expr right",
    "Binary": "Expr left, Token op, Expr right",
    "Grouping": "Expr expression",
    "Literal": "Token value",
    "Unary": "Token op, Expr right",
    "Variable" : "Token name",
}
Stmt{
    "VarDeclStmt"    : "Token name, Expr initializer",
    "ExprStmt"       : "Expr expression",
    "PrintStmt"      : "Expr expression",
    "BlockStmt"      : "std::vector<Stmt> statements",
    "IfStmt"         : "Expr condition, Stmt thenBranch, Stmt elseBranch"
}
