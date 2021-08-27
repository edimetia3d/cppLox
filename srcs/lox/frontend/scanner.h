//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <string>
#include <vector>

#include "lox/lox_error.h"
#include "lox/token/token.h"

namespace lox {
class Scanner {
 public:
  explicit Scanner(const std::string& srcs) : srcs_(&srcs) {}

  LoxError ScanAll(std::vector<Token>* output);

  LoxError ScanOne(Token* output);

  void Reset() { Reset(*srcs_); }

  void Reset(const std::string& srcs) {
    Scanner tmp(srcs);
    std::swap(tmp, *this);
  }

 private:
  void ScanOne();

  bool Match(char expected);

  char Peek(int offseet = 0);

  void AddStringToken();

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  static bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

  static bool IsAlphaNumeric(char c) { return IsAlpha(c) || IsDigit(c); }

  void AddNumToken();

  void AddToken(TokenType type);

  bool IsAtEnd() { return current_lex_pos_ >= srcs_->size(); }

  char Advance();

  void AddIdentifierToken();

  const std::string* srcs_;
  int start_lex_pos_ = 0;
  int current_lex_pos_ = 0;
  Token last_scan_;
  bool new_token_scaned_ = false;
  LoxError err_;
  int line_ = 0;
  void ResetTokenBeg();
  void ResetErr();
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
