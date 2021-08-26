Expr{
    "Assign"   : "Token name, Expr value",
    "Logical"  : "Expr left, Token op, Expr right",
    "Binary": "Expr left, Token op, Expr right",
    "Grouping": "Expr expression",
    "Literal": "Token value",
    "Unary": "Token op, Expr right",
    "Call" : "Expr callee, Token paren, std::vector<Expr> arguments",
    "GetAttr"  : "Expr src_object, Token attr_name",
    "SetAttr"  : "Expr src_object, Token attr_name, Expr value",
    "Variable" : "Token name",
}
Stmt{
    "VarDecl"    : "Token name, Expr initializer",
    "While"      : "Expr condition, Stmt body",
    "Expr"       : "Expr expression",
    "Function"   : "Token name, std::vector<Token> params, std::vector<Stmt> body",
    "Class"      : "Token name, Expr superclass, std::vector<Stmt> methods",
    "Print"      : "Expr expression",
    "Return"     : "Token keyword, Expr value",
    "Block"      : "std::vector<Stmt> statements",
    "If"         : "Expr condition, Stmt thenBranch, Stmt elseBranch",
    "Break"      : "Token src_token",
}
