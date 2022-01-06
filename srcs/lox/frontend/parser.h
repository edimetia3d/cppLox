//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_PARSER_H_
#define CPPLOX_INCLUDES_LOX_PARSER_H_

#include <iostream>
#include <memory>
#include <vector>

#include "lox/ast/ast.h"
#include "lox/lox_error.h"
#include "lox/token/token.h"

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

  Stmt AnyStatement();

  Stmt BlockStmt();
  Stmt BreakStmt();
  Stmt ClassDefStmt();
  Stmt ExpressionStmt();
  Stmt ForStmt();
  Stmt FunStmt(const std::string& kind);
  Stmt IfStmt();
  Stmt PrintStmt();
  Stmt ReturnStmt();
  Stmt VarDefStmt();
  Stmt WhileStmt();

  std::vector<Stmt> GetBlocks();

  Expr AnyExpression() { return AssignExpr(); }
  Expr AssignExpr();

  template <Expr (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... MATCH_TYPES>
  Expr BinaryExpr();
  Expr OrExpr();

  Expr AndExpr();

  Expr EqualityExpr();

  Expr ComparisonExpr();

  Expr TermExpr();

  Expr FactorExpr();

  Expr UnaryExpr();

  Expr CallExpr();

  Expr FinishCall(const Expr& callee);

  Expr Primary();
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_PARSER_H_
