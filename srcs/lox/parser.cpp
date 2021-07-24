//
// License: MIT
//

#include "lox/parser.h"
lox::ExprPointer lox::Parser::Equality() {
  return BinaryExpression<&Parser::Comparison, TokenType::BANG_EQUAL,
                          TokenType::EQUAL_EQUAL>();
}
lox::ExprPointer lox::Parser::Comparison() {
  return BinaryExpression<&Parser::Term, TokenType::GREATER,
                          TokenType::GREATER_EQUAL, TokenType::LESS,
                          TokenType::LESS_EQUAL>();
}
lox::ExprPointer lox::Parser::Term() {
  return BinaryExpression<&Parser::Factor, TokenType::MINUS, TokenType::PLUS>();
}
lox::ExprPointer lox::Parser::Factor() {
  return BinaryExpression<&Parser::Unary, TokenType::SLASH, TokenType::STAR>();
}
lox::ExprPointer lox::Parser::Unary() {
  if (AdvanceIfMatchAny<TokenType::BANG, TokenType::MINUS>()) {
    Token op = Previous();
    auto right = this->Unary();
    return ExprPointer(new lox::Unary(op, right));
  }
  return Primary();
}
lox::ExprPointer lox::Parser::Primary() {
  if (AdvanceIfMatchAny<TokenType::FALSE, TokenType::TRUE, TokenType::NIL,
                        TokenType::NUMBER, TokenType::STRING>())
    return ExprPointer(new Literal(Previous()));

  if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
    auto expr = Expression();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
    return ExprPointer(new Grouping(expr));
  }
  throw Error(Peek(), "Primary get unknown token");
}
lox::Token lox::Parser::Consume(lox::TokenType type,
                                const std::string& message) {
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
lox::ExprPointer lox::Parser::Parse() {
  try {
    return Expression();
  } catch (ParserException& exception) {
    return ExprPointer(nullptr);
  }
}
