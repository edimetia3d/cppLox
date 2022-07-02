//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_PARSER_H_
#define CPPLOX_INCLUDES_LOX_PARSER_H_

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "lox/ast/ast.h"
#include "lox/common/lox_error.h"
#include "lox/frontend/scanner.h"
#include "lox/token/token.h"
namespace lox {

class ParserError : public LoxErrorWithExitCode<EX_DATAERR> {
 public:
  using LoxErrorWithExitCode<EX_DATAERR>::LoxErrorWithExitCode;
};

enum class ParserType {
  RECURSIVE_DESCENT,
  PRATT_PARSER,
};

/**
 * Parser will error only when parsing cannot be continued. In other word, only Syntactic error will prevent ast
 * generation, all semantic error will be delayed to semantic check.
 *
 * Eg: if a break statement were found in some invalid place, the parsing will continue and no error would be thrown.
 */
class Parser {
 public:
  static std::shared_ptr<Parser> Make(ParserType type, Scanner* scanner);
  static std::shared_ptr<Parser> Make(std::string type, Scanner* scanner);
  explicit Parser(Scanner* scanner) : scanner_(scanner) { current = scanner_->ScanOne(); }

  std::unique_ptr<lox::Module> Parse();

 protected:
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

  virtual ExprPtr AnyExpression() = 0;

  std::vector<StmtPtr> GetBlocks();

  Scanner* scanner_;
  Token previous;
  Token current;
  bool err_found = false;
};

class ParserWithExprUtils : public Parser {
 public:
  using Parser::Parser;
  ExprPtr ParseCallExpr(ExprPtr expr);
  ExprPtr ParseAssignOrSetAttr(ExprPtr left_expr, ExprPtr right_expr, Token equal_token);
  ExprPtr ParseGetItemExpr(ExprPtr left_side);
  ExprPtr ForceCommaExpr();
  ExprPtr ParseTensorExpr();
};

class RecursiveDescentParser : public ParserWithExprUtils {
 public:
  using ParserWithExprUtils::ParserWithExprUtils;

 protected:
  ExprPtr AnyExpression() override;
  ExprPtr CommaExpr();
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

  template <ExprPtr (RecursiveDescentParser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... MATCH_TYPES>
  ExprPtr BinaryExpr();
};

/**
 * The int value of InfixPrecedence will be used in comparison, the order or the enum item is very important.
 */
enum class InfixPrecedence {
  LOWEST = 0,            // should not used
  COMMA,                 // ,
  ASSIGNMENT,            // =
  OR,                    // or
  AND,                   // and
  EQUALITY,              // == !=
  COMPARISON,            // < > <= >=
  TERM,                  // + -
  FACTOR,                // * /
  UNARY,                 // ! -
  CALL_OR_DOT_OR_INDEX,  // . () []
};

/**
 * Note:
 * 1. All the PrefixExpr are RIGHT_TO_LEFT associative.
 * 2. RIGHT_TO_LEFT is implemented by recursion.
 */
enum class InfixAssociativity {
  LEFT_TO_RIGHT,
  RIGHT_TO_LEFT,
};

struct InfixOpInfoMap {
  struct InfixOpInfo {
    InfixPrecedence precedence;
    InfixAssociativity associativity;
  };

  static InfixOpInfoMap& Instance();

  static InfixOpInfo* Get(TokenType type);

  static InfixOpInfo* Get(Token token) { return Get(token->type); }

  std::map<TokenType, InfixOpInfo> data;

 private:
  InfixOpInfoMap();
};

class PrattParser : public ParserWithExprUtils {
 public:
  using ParserWithExprUtils::ParserWithExprUtils;

 protected:
  ExprPtr AnyExpression() override { return DoAnyExpression(); }
  ExprPtr DoAnyExpression(InfixPrecedence lower_bound = InfixPrecedence::LOWEST);
  ExprPtr PrefixExpr();
  ExprPtr InfixExpr(ExprPtr left_side_expr);
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_PARSER_H_
