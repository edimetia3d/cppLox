//
// License: MIT
//

#include "lox/parser.h"

#include <iostream>
namespace lox {
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
  return Primary();
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
      std::cout << error.what() << std::endl;
      Synchronize();
      auto remained_statments = Parse();
      statements.insert(statements.end(), remained_statments.begin(), remained_statments.end());
    }
  }

  return statements;
}
lox::Stmt lox::Parser::Statement() {
  if (AdvanceIfMatchAny<TokenType::PRINT>()) {
    return PrintStmtStatement();
  }

  return ExprStmtStatement();
}
Stmt Parser::PrintStmtStatement() {
  Expr value = Expression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return Stmt(new PrintStmtState(value));
}
Stmt Parser::ExprStmtStatement() {
  Expr expr = Expression();
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
    return Stmt(new ExprStmtState(expr));
  }
  { return Stmt(new PrintStmtState(expr)); }
}
Stmt Parser::Declaration() {
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
  Expr expr = Equality();

  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    Token equals = Previous();
    Expr value = Assignment();  // use recurse to impl the right-associative

    if (auto state = expr.DownCastState<VariableState>()) {
      Token name = state->name;
      return Expr(new AssignState(name, value));
    }

    throw Error(equals, "Invalid assignment target.");
  }

  return expr;
}
}  // namespace lox
