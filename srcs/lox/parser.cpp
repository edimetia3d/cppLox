//
// License: MIT
//

#include "lox/parser.h"
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
    statements.push_back(Declaration());
  }

  return statements;
}
lox::Stmt lox::Parser::Statement() {
  if (AdvanceIfMatchAny<TokenType::PRINT>()) {
    return PrintStatement();
  }

  return ExpressionStatement();
}
Stmt Parser::PrintStatement() {
  Expr value = Expression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return Stmt(new PrintState(value));
}
Stmt Parser::ExpressionStatement() {
  Expr expr = Expression();
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
    return Stmt(new ExpressionState(expr));
  }
  { return Stmt(new PrintState(expr)); }
}
Stmt Parser::Declaration() {
  try {
    if (AdvanceIfMatchAny<TokenType::VAR>()) {
      auto name = Consume(TokenType::IDENTIFIER, "Expect IDENTIFIER after var decl.");
      Expr init_expr;
      if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
        init_expr = Expression();
      }
      Consume(TokenType::SEMICOLON, "Expect IDENTIFIER after var decl.");
      return Stmt(new VarState(name, init_expr));
    }

    return Statement();
  } catch (ParserException& error) {
    Synchronize();
    return Stmt(nullptr);
  }
}
}  // namespace lox
