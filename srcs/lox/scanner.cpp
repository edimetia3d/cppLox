//
// License: MIT
//

#include "scanner.h"
std::vector<Token> Scanner::Scan() {
  std::vector<Token> ret;
  ret.emplace_back(srcs_);
  return ret;
}
const std::string& Token::Str() { return str_; }
