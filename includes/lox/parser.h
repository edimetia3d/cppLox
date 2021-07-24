//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_PARSER_H_
#define CPPLOX_INCLUDES_LOX_PARSER_H_

#include <error.h>
#include <lox/error.h>

#include <memory>
#include <vector>

#include "lox/ast_nodes/expr.h"
#include "lox/token.h"
namespace lox {

class Parser;
template <TokenType type, TokenType... remained_types>
struct TokenRecursiveMatch {
  static bool Run(Parser* p) {
    if (!TokenRecursiveMatch<type>::Run(p)) {
      return TokenRecursiveMatch<remained_types...>::Run(p);
    } else {
      return true;
    }
  }
};

template <TokenType type>
struct TokenRecursiveMatch<type> {
  static bool Run(Parser* p);
};

class Parser {
 public:
  explicit Parser(const std::vector<Token>& tokens) : tokens(tokens) {}

  ExprPointer Parse();

 private:
  const std::vector<Token>& tokens;
  int current_idx = 0;

  struct ParserException : public std::exception {
    explicit ParserException(const Error& err) : err(err) {}
    const char* what() noexcept {
      static std::string last_err;
      last_err = err.Message().c_str();
      return last_err.c_str();
    }

   private:
    Error err;
  };

  const Token& Peek() { return tokens[current_idx]; }

  Token Consume(TokenType type, const std::string& message);

  ParserException Error(Token token, const std::string& msg) {
    return ParserException(lox::Error(token, msg));
  }

  bool IsAtEnd() { return Peek().type_ == TokenType::EOF_TOKEN; }

  const Token& Previous() { return tokens[current_idx - 1]; }

  const Token& Advance() {
    if (!IsAtEnd()) {
      ++current_idx;
    }
    return Previous();
  }

  bool Check(TokenType type) {
    if (IsAtEnd()) return false;
    return Peek().type_ == type;
  }

  void Synchronize();

  template <TokenType type, TokenType... remained_types>
  friend struct TokenRecursiveMatch;

  /**
   * If current token match any of types, return True **and** move cursor to
   * next token
   */
  template <TokenType... types>
  inline bool AdvanceIfMatchAny() {
    return TokenRecursiveMatch<types...>::Run(this);
  }

  ExprPointer Expression() { return Equality(); }

  template <ExprPointer (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(),
            TokenType... MATCH_TYPES>
  ExprPointer BinaryExpression() {
    // This function is the impl of  " left_expr (<binary_op> right_expr)* "

    // All token before this->current has been parsed to the expr
    auto expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();

    // if this->current is matched , we should parse all tokens after
    // this->current into a r_expr, because repeating is allowed, we do it
    // multiple times
    // if this->current is not matched, we could just return the expr
    while (AdvanceIfMatchAny<MATCH_TYPES...>()) {
      Token op = Previous();
      auto r_expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();
      expr = ExprPointer(new Binary(expr, op, r_expr));
    }
    // ok now it's done
    return expr;
  }

  ExprPointer Equality();

  ExprPointer Comparison();

  ExprPointer Term();

  ExprPointer Factor();

  ExprPointer Unary();

  ExprPointer Primary();
};

template <TokenType type>
bool TokenRecursiveMatch<type>::Run(Parser* p) {
  if (p->Check(type)) {
    p->Advance();
    return true;
  };
  return false;
}
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_PARSER_H_
