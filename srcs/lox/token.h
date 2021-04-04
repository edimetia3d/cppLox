//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_TOKEN_H_
#define CPPLOX_SRCS_LOX_TOKEN_H_

#include <string>

namespace lox {

// clang-format off
enum class TokenType {
  // Single-character tokens.
  LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
  COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR,

  // One or two character tokens.
  BANG, BANG_EQUAL,
  EQUAL, EQUAL_EQUAL,
  GREATER, GREATER_EQUAL,
  LESS, LESS_EQUAL,

  // Literals.
  IDENTIFIER, STRING, NUMBER,

  // Keywords.
  AND, CLASS, ELSE, FALSE, FUN, FOR, IF, NIL, OR,
  PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE,

  EOF_TOKEN
};
// clang-format on

class Token {
 public:
  explicit Token(TokenType type, std::string lexeme, int line)
      : type_(type), line_(line), lexeme_(std::move(lexeme)) {}
  std::string Str();

 private:
  std::string lexeme_;
  TokenType type_;
  int line_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_TOKEN_H_
