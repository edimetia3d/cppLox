//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_H_

#include <memory>
#include <string>

#include "lox/common/location.h"
#include "lox/common/lox_error.h"
#include "lox/common/string_ref.h"
#include "lox/token/token_type.h"

namespace lox {

class TokenState {
public:
  std::string Dump() const;

  RefString lexeme;
  TokenType type;
  Location location;

protected:
  friend class Token;
  explicit TokenState(TokenType type, RefString lexeme, Location location)
      : lexeme(std::move(lexeme)), type(type), location(std::move(location)) {}
};

class Token {
public:
  Token() = default;
  Token(TokenType type, RefString lexeme, Location location) {
    state_ = std::shared_ptr<TokenState>(new TokenState(type, lexeme, std::move(location)));
  }

  static TokenType GetIdentifierType(const std::string &identifier);

  TokenState *operator->() const { return state_.get(); }
  operator bool() const { return static_cast<bool>(state_); }

private:
  Token(std::shared_ptr<TokenState> state) : state_(state) {}
  std::shared_ptr<TokenState> state_;
};
// Token is designed to be shared ptr like, and only contains one data member
static_assert(sizeof(Token) == sizeof(std::shared_ptr<TokenState>));

} // namespace lox
#endif // CPPLOX_INCLUDES_LOX_TOKEN_H_
