//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <string>
#include <vector>

#include "lox/common/lox_error.h"
#include "lox/token/token.h"

namespace lox {

class ScannerError : public LoxErrorWithExitCode<EX_DATAERR> {
  using LoxErrorWithExitCode<EX_DATAERR>::LoxErrorWithExitCode;
};

class Scanner {
 public:
  explicit Scanner(const std::string& srcs, std::string src_file_name)
      : srcs_(&srcs), src_file_name(std::move(src_file_name)) {}

  std::vector<Token> ScanAll();

  Token ScanOne();

  void Reset() { Reset(*srcs_); }

  void Reset(const std::string& srcs) {
    Scanner tmp(srcs, src_file_name);
    std::swap(tmp, *this);
  }

 private:
  bool MatchAndAdvance(char expected);

  char Peek(int offseet = 0);

  Token AddStringToken();

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  static bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

  static bool IsAlphaNumeric(char c) { return IsAlpha(c) || IsDigit(c); }

  Token AddNumToken();

  Token AddToken(TokenType type);

  void Error(const std::string& msg);

  bool IsAtEnd() { return current_lex_pos_ >= srcs_->size(); }
  char LastChar();

  char Advance();

  void StartNewLine() {
    line_++;
    col_ = 0;
  }

  Token AddIdentifierToken();

  const std::string* srcs_;
  int start_lex_pos_ = 0;
  int current_lex_pos_ = 0;
  int line_ = 0;
  int col_ = 0;
  std::string src_file_name;
  void ResetTokenBeg();
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_SCANNER_H_
