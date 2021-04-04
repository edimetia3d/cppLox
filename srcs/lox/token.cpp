//
// License: MIT
//

#include "lox/token.h"

#include <map>
namespace lox {
// clang-format off
static std::map<std::string,TokenType> g_reserved_map{
    {"and",    TokenType::AND},
    {"class",  TokenType::CLASS},
    {"else",   TokenType::ELSE},
    {"false",  TokenType::FALSE},
    {"for",    TokenType::FOR},
    {"fun",    TokenType::FUN},
    {"if",     TokenType::IF},
    {"nil",    TokenType::NIL},
    {"or",     TokenType::OR},
    {"print",  TokenType::PRINT},
    {"return", TokenType::RETURN},
    {"super",  TokenType::SUPER},
    {"this",   TokenType::THIS},
    {"true",   TokenType::TRUE},
    {"var",    TokenType::VAR},
    {"while",  TokenType::WHILE}
};
// clang-format on

std::string Token::Str() const {
  return std::string("Type: ") + std::to_string((int)type_) +
         " Lexme: " + lexeme_ + " @Line: " + std::to_string(line_);
}

TokenType Token::GetIdentifierType(const std::string& identifier) {
  if (g_reserved_map.count(identifier) == 0) {
    return TokenType::IDENTIFIER;
  } else {
    return g_reserved_map[identifier];
  }
}
}  // namespace lox