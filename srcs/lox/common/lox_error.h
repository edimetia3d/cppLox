//
// LICENSE: MIT
//

#ifndef LOX_INCLUDES_LOX_LOX_ERROR_H_
#define LOX_INCLUDES_LOX_LOX_ERROR_H_
#include <memory>
#include <string>
#include <vector>

#include "sysexits.h"

namespace lox {
/**
 * Our implementation uses a c++ style error handling system. We throw an LoxError when we encounter something that
 * can't be handled.
 */
class LoxError : public std::exception {
public:
  LoxError() = default;
  explicit LoxError(const std::string &what) : what_(what){};
  explicit LoxError(std::string &&what) : what_(std::move(what)){};

  const char *what() const noexcept override;

  uint8_t exit_code = -1; // if error should cause exit, this should be used.
protected:
  mutable std::string what_;
};
} // namespace lox

template <uint8_t DEFAULT_EXIT_CODE> class LoxErrorWithExitCode : public lox::LoxError {
public:
  LoxErrorWithExitCode() = default;
  explicit LoxErrorWithExitCode(const std::string &what) : LoxError(what) { exit_code = DEFAULT_EXIT_CODE; };
  explicit LoxErrorWithExitCode(std::string &&what) : LoxError(std::move(what)) { exit_code = DEFAULT_EXIT_CODE; };
};

#endif // LOX_INCLUDES_LOX_LOX_ERROR_H_
