//
// License: MIT
//

#include "token.h"
namespace lox {
std::string Token::Str() {
  return std::string("Type: ") + std::to_string((int)type_) +
         " Lexme: " + lexeme_ + " @Line: " + std::to_string(line_);
}
}  // namespace lox