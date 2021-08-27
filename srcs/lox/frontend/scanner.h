//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <string>
#include <vector>

#include "lox/backend/tree_walker/error.h"
#include "token.h"

namespace lox {
class Scanner {
 public:
  explicit Scanner(const std::string& srcs) : srcs_(&srcs) {}

  Error ScanAll(std::vector<Token>* output);

  Error ScanOne(Token* output);

  void Reset() {
    Scanner tmp(*srcs_);
    std::swap(tmp, *this);
  }

  void Reset(const std::string& srcs) {
    Scanner tmp(srcs);
    std::swap(tmp, *this);
  }

 private:
  void TryScanOne();

  bool Match(char expected);

  char Peek(int offseet = 0);

  void AddStringToken();

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  static bool IsAlpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  }

  static bool IsAlphaNumeric(char c) { return IsAlpha(c) || IsDigit(c); }

  void AddNumToken();

  void AddToken(TokenType type);

  bool IsAtEnd() { return current_lex_pos_ >= srcs_->size(); }

  char Advance();

  void AddIdentifierToken();

  const std::string* srcs_;
  int start_lex_pos_ = 0;
  int current_lex_pos_ = 0;
  std::unique_ptr<Token> last_scan_;
  int line_ = 0;
  Error err_;
  void ResetTokenBeg();
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
