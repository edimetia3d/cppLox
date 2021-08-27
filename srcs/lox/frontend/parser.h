//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_PARSER_H_
#define CPPLOX_INCLUDES_LOX_PARSER_H_

#include <iostream>
#include <memory>
#include <vector>

#include "lox/frontend/ast/ast.h"
#include "lox/frontend/token.h"
#include "lox/lox_error.h"

namespace lox {

class Parser {
 public:
  explicit Parser(const std::vector<Token>& tokens) : tokens(tokens) {}

  std::vector<Stmt> Parse();

 private:
  const std::vector<Token>& tokens;
  int current_idx = 0;
  bool err_found = false;

  const Token& Peek() { return tokens[current_idx]; }

  Token Consume(TokenType type, const std::string& message);

  LoxError Error(Token token, const std::string& msg);

  bool IsAtEnd() { return Peek()->type == TokenType::EOF_TOKEN; }

  const Token& Previous() { return tokens[current_idx - 1]; }

  const Token& Advance() {
    if (!IsAtEnd()) {
      ++current_idx;
    }
    return Previous();
  }

  bool Check(TokenType type) {
    if (IsAtEnd()) return false;
    return Peek()->type == type;
  }

  void Synchronize();

  /**
   * If current token match any of types, return True **and** move cursor to
   * next token
   */
  template <TokenType type, TokenType... remained_types>
  bool AdvanceIfMatchAny();
  Stmt Declaration();
  Stmt FunctionDef(const std::string& kind);
  Stmt Statement();
  Stmt Print();
  Stmt Return();
  Stmt While();
  Stmt Break();
  Stmt ForStmtSugar();
  Stmt ExpressionStmt();
  Stmt Block();
  std::vector<Stmt> GetBlocks();
  Stmt If();
  Expr ExpressionExpr() { return Assignment(); }
  Expr Assignment();

  template <Expr (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... MATCH_TYPES>
  Expr BinaryExpression();
  Expr Or();

  Expr And();

  Expr Equality();

  Expr Comparison();

  Expr Term();

  Expr Factor();

  Expr Unary();

  Expr Call();

  Expr FinishCall(const Expr& callee);

  Expr Primary();
  Stmt ClassDef();
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_PARSER_H_
