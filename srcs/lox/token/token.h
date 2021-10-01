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

class TokenBase;
using Token = std::shared_ptr<TokenBase>;

class TokenBase {
 public:
  static Token Make(const TokenType& type, const std::string& lexeme, int line) {
    return std::shared_ptr<TokenBase>(new TokenBase(type, lexeme, line));
  }

  static TokenType GetIdentifierType(const std::string& identifier);
  std::string Str() const;

  std::string lexeme;
  TokenType type;
  int line;

 protected:
  explicit TokenBase(const TokenType& type, const std::string& lexeme, int line)
      : type(type), line(line), lexeme(std::move(lexeme)) {}
};

static inline Token MakeToken(const TokenType& type, const std::string& lexeme, int line) {
  return TokenBase::Make(type, lexeme, line);
}

/**
 * Note: only class LoxInterpreter will handle Errors, all other class only generate
 * errors, and return `PrefixTokenError` to the caller.
 */
template <class CRTP>  // use crtp to make different derived class
class PrefixTokenError : public LoxError {
 public:
  explicit PrefixTokenError(const Token& token, const std::string& message)
      : LoxError(CRTP::StrName() + token->Str() + " what(): " + message) {}
  [[nodiscard]] const Token& SourceToken() const { return token_; }

 private:
  Token token_ = MakeToken(TokenType::_TOKEN_COUNT_NUMBER, "None", -1);
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_TOKEN_H_
