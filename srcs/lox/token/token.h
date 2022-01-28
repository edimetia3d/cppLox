//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_H_

#include <memory>
#include <string>

#include "lox/lox_error.h"
#include "token_type.h"

namespace lox {

class TokenState {
 public:
  std::string Dump() const;

  std::string lexeme;
  TokenType type;
  int line;

 protected:
  friend class Token;
  explicit TokenState(const TokenType& type, const std::string& lexeme, int line)
      : type(type), line(line), lexeme(std::move(lexeme)) {}
};

struct Token {
  Token() = default;
  Token(const TokenType& type, const std::string& lexeme, int line) {
    state_ = std::shared_ptr<TokenState>(new TokenState(type, lexeme, line));
  }

  static TokenType GetIdentifierType(const std::string& identifier);

  TokenState* operator->() const { return state_.get(); }
  operator bool() const { return static_cast<bool>(state_); }

 private:
  Token(std::shared_ptr<TokenState> state) : state_(state) {}
  std::shared_ptr<TokenState> state_;
};
// Token is designed to be shared ptr like, and only contains one data member
static_assert(sizeof(Token) == sizeof(std::shared_ptr<TokenState>));

/**
 * Note: only class LoxInterpreter will handle Errors, all other class only generate
 * errors, and return `PrefixTokenError` to the caller.
 */
template <class CRTP>  // use crtp to make different derived class
class PrefixTokenError : public LoxError {
 public:
  explicit PrefixTokenError(const Token& token, const std::string& message)
      : LoxError(CRTP::StrName() + token->Dump() + " what(): " + message) {}
  [[nodiscard]] const Token& SourceToken() const { return token_; }

 private:
  Token token_ = Token(TokenType::_TOKEN_COUNT_NUMBER, "None", -1);
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_TOKEN_H_
