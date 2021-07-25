//
// LICENSE: MIT
//

#include "lox_object.h"

namespace lox {
namespace object {
LoxObjectPointer lox::object::LoxObject::FromLiteralToken(const Token& token) {
  switch (token.type_) {
    case TokenType::NUMBER:
      return LoxObjectPointer(new Number(std::stod(token.lexeme_)));
    case TokenType::STRING:
      return LoxObjectPointer(new String(token.lexeme_));
    default:
      throw "Not Valid Literal";
  }
}
}  // namespace object
}  // namespace lox
