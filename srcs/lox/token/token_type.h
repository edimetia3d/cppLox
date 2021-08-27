//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_TYPE_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_TYPE_H_
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
  PRINT, RETURN, THIS, TRUE, VAR, WHILE,BREAK,CONTINUE,

  EOF_TOKEN,

  _TOKEN_COUNT_NUMBER
};
// clang-format on
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_TOKEN_TYPE_H_
