//
// License: MIT
//

#include "scanner.h"
namespace lox {
std::vector<Token> Scanner::Scan() {
  std::vector<Token> ret;
  ret.emplace_back(srcs_);
  return ret;
}
}  // namespace lox