//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <string>
#include <vector>

#include "lox/token.h"

namespace lox {
class Scanner {
 public:
  Scanner(const std::string& srcs) : srcs_(srcs) {}

  std::vector<Token> Scan();

 private:
  const std::string& srcs_;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
