//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_H_

#include <memory>
#include <string>

#include "lox/common/lox_error.h"
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

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_TOKEN_H_
