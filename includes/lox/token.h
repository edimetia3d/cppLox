//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_H_

#include <string>

#include "lox/token_type.h"

namespace lox {

class Token {
 public:
  static TokenType GetIdentifierType(const std::string& identifier);
  explicit Token(TokenType type, std::string lexeme, int line)
      : type_(type), line_(line), lexeme_(std::move(lexeme)) {}
  std::string Str() const;

  std::string lexeme_;
  TokenType type_;
  int line_;
};
}  // namespace lox

#endif  // CPPLOX_INCLUDES_LOX_TOKEN_H_
