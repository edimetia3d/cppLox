//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <string>
#include <vector>

class Token {
 public:
  Token(const std::string& str) : str_(str) {}
  const std::string& Str();

 private:
  std::string str_;
};

class Scanner {
 public:
  Scanner(const std::string& srcs) : srcs_(srcs) {}

  std::vector<Token> Scan();

 private:
  const std::string& srcs_;
};

#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
