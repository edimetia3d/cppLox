//
// License: MIT
//

#include "lox/parser.h"

#include <iostream>

#include "lox/global_setting/global_setting.h"
namespace lox {
Expr Parser::Or() {
  auto expr = And();

  while (AdvanceIfMatchAny<TokenType::OR>()) {
    Token op = Previous();
    auto r_expr = And();
    expr = Expr(new LogicalState(expr, op, r_expr));
  }
  return expr;
}
Expr Parser::And() {
  auto expr = Equality();

  while (AdvanceIfMatchAny<TokenType::AND>()) {
    Token op = Previous();
    auto r_expr = Equality();
    expr = Expr(new LogicalState(expr, op, r_expr));
  }
  return expr;
}

lox::Expr lox::Parser::Equality() {
  return BinaryExpression<&Parser::Comparison, TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL>();
}
lox::Expr lox::Parser::Comparison() {
  return BinaryExpression<&Parser::Term, TokenType::GREATER, TokenType::GREATER_EQUAL, TokenType::LESS,
                          TokenType::LESS_EQUAL>();
}
lox::Expr lox::Parser::Term() { return BinaryExpression<&Parser::Factor, TokenType::MINUS, TokenType::PLUS>(); }
lox::Expr lox::Parser::Factor() { return BinaryExpression<&Parser::Unary, TokenType::SLASH, TokenType::STAR>(); }
lox::Expr lox::Parser::Unary() {
  if (AdvanceIfMatchAny<TokenType::BANG, TokenType::MINUS>()) {
    Token op = Previous();
    auto right = this->Unary();
    return Expr(new lox::UnaryState(op, right));
  }
  return Call();
}
lox::Expr lox::Parser::Primary() {
  if (AdvanceIfMatchAny<TokenType::FALSE, TokenType::TRUE, TokenType::NIL, TokenType::NUMBER, TokenType::STRING>())
    return Expr(new LiteralState(Previous()));

  if (AdvanceIfMatchAny<TokenType::IDENTIFIER>()) {
    return Expr(new VariableState(Previous()));
  }

  if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
    auto expr = Expression();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
    return Expr(new GroupingState(expr));
  }
  throw Error(Peek(), "Primary get unknown token");
}
lox::Token lox::Parser::Consume(lox::TokenType type, const std::string& message) {
  if (Check(type)) return Advance();
  throw Error(Peek(), message);
}
void lox::Parser::Synchronize() {
  Advance();

  while (!IsAtEnd()) {
    if (Previous().type_ == TokenType::SEMICOLON) return;

    switch (Peek().type_) {
      case TokenType::CLASS:
      case TokenType::FUN:
      case TokenType::VAR:
      case TokenType::FOR:
      case TokenType::IF:
      case TokenType::WHILE:
      case TokenType::PRINT:
      case TokenType::RETURN:
        return;
    }

    Advance();
  }
}
std::vector<lox::Stmt> lox::Parser::Parse() {
  std::vector<lox::Stmt> statements;
  while (!IsAtEnd()) {
    try {
      auto stmt = Declaration();
      statements.push_back(stmt);
    } catch (ParserError& error) {
      Synchronize();
      Parse();
    }
  }
  if (err_found) {
    return {};
  } else {
    return statements;
  }
}
lox::Stmt lox::Parser::Statement() {
  if (AdvanceIfMatchAny<TokenType::IF>()) return IfStmt();
  if (AdvanceIfMatchAny<TokenType::WHILE>()) return WhileStmt();
  if (AdvanceIfMatchAny<TokenType::BREAK, TokenType::CONTINUE>()) return BreakStmt();
  if (AdvanceIfMatchAny<TokenType::FOR>()) return ForStmtSugar();
  if (AdvanceIfMatchAny<TokenType::PRINT>()) return PrintStmt();
  if (AdvanceIfMatchAny<TokenType::RETURN>()) return ReturnStmt();
  if (AdvanceIfMatchAny<TokenType::LEFT_BRACE>()) return BlockStmt();

  return ExprStmt();
}
Stmt Parser::PrintStmt() {
  Expr value = Expression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return Stmt(new PrintStmtState(value));
}
Stmt Parser::ExprStmt() {
  Expr expr = Expression();
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
    return Stmt(new ExprStmtState(expr));
  } else {
    if (GlobalSetting().interactive_mode) {
      return Stmt(new PrintStmtState(expr));
    } else {
      Error(Peek(), "Non-interactive mode must have ; after expression");
    }
  }
  return Stmt(nullptr);
}
Stmt Parser::Declaration() {
  if (AdvanceIfMatchAny<TokenType::CLASS>()) return ClassDef();
  if (AdvanceIfMatchAny<TokenType::FUN>()) return FunctionDef("function");
  if (AdvanceIfMatchAny<TokenType::VAR>()) {
    auto name = Consume(TokenType::IDENTIFIER, "Expect IDENTIFIER after key `var`.");
    Expr init_expr(nullptr);
    if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
      init_expr = Expression();
    }
    Consume(TokenType::SEMICOLON, "Expect ; when var decl finish.");
    return Stmt(new VarDeclStmtState(name, init_expr));
  }

  return Statement();
}
Expr Parser::Assignment() {
  Expr expr = Or();

  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    Token equals = Previous();
    Expr value = Assignment();  // use recurse to impl the right-associative

    if (auto state = expr.DownCastState<VariableState>()) {
      Token name = state->name;
      return Expr(new AssignState(name, value));
    }

    Error(equals, "Invalid assignment target.");
  }

  return expr;
}
Stmt Parser::BlockStmt() { return Stmt(new BlockStmtState(Blocks())); }
std::vector<Stmt> Parser::Blocks() {
  std::vector<Stmt> statements;

  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    statements.push_back(Declaration());
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after block.");
  return statements;
}
Stmt Parser::IfStmt() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  Expr condition = Expression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after if condition.");

  Stmt thenBranch = Statement();
  Stmt elseBranch(nullptr);
  if (AdvanceIfMatchAny<TokenType::ELSE>()) {
    elseBranch = Statement();
  }

  return Stmt(new IfStmtState(condition, thenBranch, elseBranch));
}
Stmt Parser::WhileStmt() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  Expr condition = Expression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");
  ++while_loop_level;
  Stmt body = Statement();
  --while_loop_level;

  return Stmt(new WhileStmtState(condition, body));
}
Stmt Parser::ForStmtSugar() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'for'.");

  Stmt initializer(nullptr);
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
  } else if (Check(TokenType::VAR)) {
    initializer = Declaration();
  } else {
    initializer = ExprStmt();
  }

  Expr condition(nullptr);
  if (!Check(TokenType::SEMICOLON)) {
    condition = Expression();
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after loop condition.");

  Expr increment(nullptr);
  if (!Check(TokenType::RIGHT_PAREN)) {
    increment = Expression();
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after for clauses.");
  ++while_loop_level;
  Stmt body = Statement();
  --while_loop_level;

  if (increment.IsValid()) {
    std::vector<Stmt> body_with_increasement = {body};
    body_with_increasement.push_back(Stmt(new ExprStmtState(increment)));
    body = Stmt(new BlockStmtState(body_with_increasement));
  }
  if (!condition.IsValid()) {
    auto tmp_true_token = Token(TokenType::TRUE, "for_sugar_true", Peek().line_);
    condition = Expr(new LiteralState(tmp_true_token));
  }
  body = Stmt(new WhileStmtState(condition, body));

  if (initializer.IsValid()) {
    std::vector<Stmt> body_with_initializer = {initializer, body};
    body = Stmt(new BlockStmtState(body_with_initializer));
  }

  return body;
}
Stmt Parser::BreakStmt() {
  auto src_token = Previous();
  if (src_token.type_ == TokenType::CONTINUE) {
    Error(Previous(), " 'continue' not supported yet.");
  }
  if (while_loop_level) {
    Consume(TokenType::SEMICOLON, std::string("Expect ';' after ") + src_token.lexeme_);
    return Stmt(new BreakStmtState(src_token));
  } else {
    Error(Previous(), std::string("Nothing to ") + src_token.lexeme_);
  }
  return Stmt(nullptr);
}
Expr Parser::Call() {
  Expr expr = Primary();

  while (true) {
    if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
      expr = FinishCall(expr);
    } else if (AdvanceIfMatchAny<TokenType::DOT>()) {
      Token name = Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      expr = Expr(new GetAttrState(expr, name));
    } else {
      break;
    }
  }

  return expr;
}
Expr Parser::FinishCall(const Expr& callee) {
  std::vector<Expr> arguments;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      if (arguments.size() >= 255) {
        Error(Peek(), "Can't have more than 255 arguments.");
      }
      arguments.push_back(Expression());
    } while (AdvanceIfMatchAny<TokenType::COMMA>());
  }

  Token paren = Consume(TokenType::RIGHT_PAREN, "Expect ')' after arguments.");

  return Expr(new CallState(callee, paren, arguments));
}
Stmt Parser::FunctionDef(const std::string& kind) {
  ++func_def_level;
  Token name = Consume(TokenType::IDENTIFIER, "Expect " + kind + " name.");
  Consume(TokenType::LEFT_PAREN, "Expect '(' after " + kind + " name.");
  std::vector<Token> parameters;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      if (parameters.size() >= 255) {
        Error(Peek(), "Can't have more than 255 parameters.");
      }

      parameters.push_back(Consume(TokenType::IDENTIFIER, "Expect parameter name."));
    } while (AdvanceIfMatchAny<TokenType::COMMA>());
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after parameters.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before " + kind + " body.");
  std::vector<Stmt> body = Blocks();
  --func_def_level;
  return Stmt(new FunctionStmtState(name, parameters, body));
}
Stmt Parser::ReturnStmt() {
  Token keyword = Previous();
  if (func_def_level) {
    Expr value(nullptr);
    if (!Check(TokenType::SEMICOLON)) {
      value = Expression();
    }

    Consume(TokenType::SEMICOLON, "Expect ';' after return value.");
    return Stmt(new ReturnStmtState(keyword, value));
  } else {
    Error(Previous(), std::string("Cannot return here."));
  }
  return Stmt(nullptr);
}
Stmt Parser::ClassDef() {
  Token name = Consume(TokenType::IDENTIFIER, "Expect class name.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before class body.");

  std::vector<Stmt> methods;
  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    methods.push_back(FunctionDef("method"));
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after class body.");

  return Stmt(new ClassStmtState(name, methods));
}

}  // namespace lox
