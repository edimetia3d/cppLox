//
// License: MIT
//

#include "token.h"

#include <map>
namespace lox {
// clang-format off
static std::map<std::string,TokenType> g_reserved_map{
    {"and",    TokenType::AND},
    {"class",  TokenType::CLASS},
    {"else",   TokenType::ELSE},
    {"false",  TokenType::FALSE_TOKEN},
    {"for",    TokenType::FOR},
    {"fun",    TokenType::FUN},
    {"if",     TokenType::IF},
    {"nil",    TokenType::NIL},
    {"or",     TokenType::OR},
    {"print",  TokenType::PRINT},
    {"return", TokenType::RETURN},
    {"this",   TokenType::THIS},
    {"true",   TokenType::TRUE_TOKEN},
    {"var",    TokenType::VAR},
    {"while",  TokenType::WHILE},
    {"break",  TokenType::BREAK},
    {"continue",  TokenType::CONTINUE},
    {"super",  TokenType::SUPER},
    {"Tensor",TokenType::TENSOR},
};
// clang-format on

std::string TokenState::Dump() const {
  return std::string("[Type:{") + std::to_string((int)type) + "},Lexme:{" + lexeme + "},@Line:" + std::to_string(line) +
         "]";
}

TokenType Token::GetIdentifierType(const std::string& identifier) {
  if (g_reserved_map.count(identifier) == 0) {
    return TokenType::IDENTIFIER;
  } else {
    return g_reserved_map[identifier];
  }
}
}  // namespace lox
