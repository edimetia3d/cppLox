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
  std::string Message();

  int ToErrCode();
  void Append(const Error &new_err);
  const Token &SourceToken() const { return token_; }

 private:
  Token token_{TokenType::_TOKEN_COUNT_NUMBER, "None", -1};
  std::string message_;
  std::string RecursiveMessage(int level);
  std::vector<ErrorNode> sub_errors;
};

struct RuntimeError : public std::exception {
  explicit RuntimeError(Error err) : err(std::move(err)) {}
  const char *what() noexcept {
    static std::string last_err;
    last_err = std::string("RuntimeError: ") + err.Message().c_str();
    return last_err.c_str();
  }

  Error err;
};

struct ParserError : public std::exception {
  explicit ParserError(const Error &err) : err(err) {}
  const char *what() noexcept {
    static std::string last_err;
    last_err = std::string("PaserError: ") + err.Message().c_str();
    return last_err.c_str();
  }

  lox::Error err;
};

#define ERR_STR(STR) Error(std::string("[") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] " + STR)
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
