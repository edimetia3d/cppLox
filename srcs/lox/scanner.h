//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <lox/error.h>

#include <string>
#include <vector>

#include "lox/token.h"

namespace lox {
class Scanner {
 public:
  explicit Scanner(const std::string& srcs) : srcs_(&srcs) {}

  Error Scan();

  const std::vector<Token>& Tokens() { return tokens_; }
  void Reset() {
    Scanner tmp(*srcs_);
    std::swap(tmp, *this);
  }

  void Reset(const std::string& srcs) {
    Scanner tmp(srcs);
    std::swap(tmp, *this);
  }

 private:
  void scanSinlge();

  bool match(char expected);

  char peek();

  void AddToken(TokenType type);

  bool isAtEnd() { return current_lex_pos_ >= srcs_->size(); }

  char Advance();
  const std::string* srcs_;
  int start_lex_pos_ = 0;
  int current_lex_pos_ = 0;
  std::vector<Token> tokens_;
  int line_ = 0;
  Error err_;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
