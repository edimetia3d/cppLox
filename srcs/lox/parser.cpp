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
      Parse();
      statements.clear();
    }
  }

  return statements;
}
lox::Stmt lox::Parser::Statement() {
  if (AdvanceIfMatchAny<TokenType::IF>()) return IfStmt();
  if (AdvanceIfMatchAny<TokenType::WHILE>()) return WhileStmt();
  if (AdvanceIfMatchAny<TokenType::FOR>()) return ForStmtSugar();
  if (AdvanceIfMatchAny<TokenType::PRINT>()) return PrintStmt();
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
      throw Error(Peek(), "Non-interactive mode must have ; after expression");
    }
  }
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
  Expr expr = Or();

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
  Stmt body = Statement();

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
  Stmt body = Statement();

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

}  // namespace lox
