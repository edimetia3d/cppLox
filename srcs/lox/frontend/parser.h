//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_PARSER_H_
#define CPPLOX_INCLUDES_LOX_PARSER_H_

#include <iostream>
#include <memory>
#include <vector>

#include "lox/ast/ast.h"
#include "lox/common/lox_error.h"
#include "lox/frontend/scanner.h"
#include "lox/token/token.h"
namespace lox {

class ParserError : public LoxError {
 public:
  using LoxError::LoxError;
};

/**
 * Parser will error only when parsing cannot be continued. In other word, only Syntactic error will prevent ast
 * generation, all semantic error will be delayed to semantic check.
 *
 * Eg: if a break statement were found in some invalid place, the parsing will continue and no error would be thrown.
 */
class Parser {
 public:
  explicit Parser(Scanner* scanner) : scanner_(scanner) { current = scanner_->ScanOne(); }

  std::unique_ptr<lox::FunctionStmt> Parse();

 private:
  const Token& Peek() { return current; }

  Token Consume(TokenType type, const std::string& message);

  void Error(Token token, const std::string& msg);

  bool IsAtEnd() { return Peek()->type == TokenType::EOF_TOKEN; }

  const Token& Previous() { return previous; }

  const Token& Advance() {
    if (!IsAtEnd()) {
      previous = current;
      current = scanner_->ScanOne();
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

  StmtPtr AnyStatement();
  StmtPtr DoAnyStatement();

  StmtPtr BlockStmt();
  StmtPtr BreakStmt();
  StmtPtr ClassDefStmt();
  StmtPtr ExpressionStmt();
  StmtPtr ForStmt();
  StmtPtr FunStmt(const std::string& kind);
  StmtPtr IfStmt();
  StmtPtr PrintStmt();
  StmtPtr ReturnStmt();
  StmtPtr VarDefStmt();
  StmtPtr WhileStmt();

  std::vector<StmtPtr> GetBlocks();

  ExprPtr AnyExpression();
  ExprPtr AssignExpr();
  ExprPtr OrExpr();
  ExprPtr AndExpr();
  ExprPtr EqualityExpr();
  ExprPtr ComparisonExpr();
  ExprPtr TermExpr();
  ExprPtr FactorExpr();
  ExprPtr UnaryExpr();
  ExprPtr CallExpr();
  ExprPtr Primary();

  template <ExprPtr (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... MATCH_TYPES>
  ExprPtr BinaryExpr();

  Scanner* scanner_;
  Token previous;
  Token current;
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_PARSER_H_
