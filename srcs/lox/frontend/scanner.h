//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_SCANNER_H_
#define CPPLOX_SRCS_LOX_SCANNER_H_

#include <cassert>
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
  explicit Scanner(std::shared_ptr<CharStream> input) : input_(std::move(input)) {}

  Token ScanOne();

private:
  bool MatchAndAdvance(char expected);

  char Peek(int offset = 0);

  Token AddStringToken();

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  static bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

  static bool IsAlphaNumeric(char c) { return IsAlpha(c) || IsDigit(c); }

  Token AddNumToken();

  Token AddToken(TokenType type);

  void Error(const std::string &msg);

  char Advance();

  void StartNewLine() { line_++; }

  Token AddIdentifierToken();

  std::shared_ptr<CharStream> input_;
  int start_lex_pos_ = 0;
  int line_ = 0;

  void ResetTokenBeg();
};
} // namespace lox
#endif // CPPLOX_SRCS_LOX_SCANNER_H_
