Expr{
    "Assign"   : "Token name, Expr value",
    "Logical"  : "Expr left, Token op, Expr right",
    "Binary": "Expr left, Token op, Expr right",
    "Grouping": "Expr expression",
    "Literal": "Token value",
    "Unary": "Token op, Expr right",
    "Call" : "Expr callee, Token paren, std::vector<Expr> arguments",
    "Variable" : "Token name",
}
Stmt{
    "VarDeclStmt"    : "Token name, Expr initializer",
    "WhileStmt"      : "Expr condition, Stmt body",
    "ExprStmt"       : "Expr expression",
    "FunctionStmt"   : "Token name, std::vector<Token> params, std::vector<Stmt> body",
    "ClassStmt"      : "Token name, std::vector<Stmt> methods",
    "PrintStmt"      : "Expr expression",
    "ReturnStmt"     : "Token keyword, Expr value",
    "BlockStmt"      : "std::vector<Stmt> statements",
    "IfStmt"         : "Expr condition, Stmt thenBranch, Stmt elseBranch",
    "BreakStmt"      : "Token src_token",
}
