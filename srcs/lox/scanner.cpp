//
// License: MIT
//

#include "scanner.h"
namespace lox {
std::vector<Token> Scanner::Scan() {
  std::vector<Token> ret;
  ret.emplace_back(TokenType::EOF_TOKEN, srcs_, 0);
  return ret;
}
}  // namespace lox