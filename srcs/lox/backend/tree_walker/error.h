//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/frontend/token.h"
#include "lox/lox_error.h"

namespace lox {
/**
 * Note: only class LoxInterpreter will handle Errors, all other class only generate
 * errors, and return `TreeWalkerError` to the caller.
 */
class TreeWalkerError : public LoxError {
 public:
  explicit TreeWalkerError(const Token &token, const std::string &message);
  [[nodiscard]] const Token &SourceToken() const { return token_; }

 private:
  Token token_ = MakeToken(TokenType::_TOKEN_COUNT_NUMBER, "None", -1);
};

template <size_t N>
struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

  char value[N];
};

template <StringLiteral name>
struct TemplateError : public std::exception {
  explicit TemplateError(TreeWalkerError err) : err(std::move(err)) {}
  const char *what() const noexcept override {
    static std::string last_err;
    last_err = std::string(name.value) + ": " + err.Message().c_str();
    return last_err.c_str();
  }

  TreeWalkerError err;
};

using RuntimeError = TemplateError<"RuntimeError">;
using ParserError = TemplateError<"ParserError">;
using ResolveError = TemplateError<"ResolveError">;
using SemanticError = TemplateError<"SemanticError">;
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
