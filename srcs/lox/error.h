//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/token.h"

namespace lox {
/**
 * Note: only class Lox will handle Errors, all other class only generate
 * errors, and return `Error` to the caller.
 */
class Error {
 public:
  using ErrorNode = std::shared_ptr<Error>;
  Error();
  explicit Error(const std::string &message);
  explicit Error(const Token &token, const std::string &message);
  std::string Message() const;

  int ToErrCode();
  void Append(const Error &new_err);
  const Token &SourceToken() const { return token_; }

 private:
  Token token_ = MakeToken(TokenType::_TOKEN_COUNT_NUMBER, "None", -1);
  std::string message_;
  std::string RecursiveMessage(int level) const;
  std::vector<ErrorNode> sub_errors;
};

template <size_t N>
struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

  char value[N];
};

template <StringLiteral name>
struct TemplateError : public std::exception {
  explicit TemplateError(Error err) : err(std::move(err)) {}
  const char *what() const noexcept override {
    static std::string last_err;
    last_err = std::string(name.value) + ": " + err.Message().c_str();
    return last_err.c_str();
  }

  Error err;
};

using RuntimeError = TemplateError<"RuntimeError">;
using ParserError = TemplateError<"ParserError">;
using ResolveError = TemplateError<"ResolveError">;
using SemanticError = TemplateError<"SemanticError">;

#define ERR_STR(STR) Error(std::string("[") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] " + STR)
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
