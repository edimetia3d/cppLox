//
// LICENSE: MIT
//

#ifndef LOX_INCLUDES_LOX_LOX_ERROR_H_
#define LOX_INCLUDES_LOX_LOX_ERROR_H_
#include <memory>
#include <string>
#include <vector>

#include <sysexits.h>

namespace lox {
/**
 * Our implementation uses a c++ style error handling system. We throw an LoxError when we encounter something that
 * can't be handled.
 */
class LoxError : public std::exception {
 public:
  LoxError() = default;
  explicit LoxError(const std::string& what, uint8_t exit_code = EX_SOFTWARE) : what_(what), exit_code(exit_code){};
  explicit LoxError(std::string&& what, uint8_t exit_code = EX_SOFTWARE)
      : what_(std::move(what)), exit_code(exit_code){};

  const char* what() const noexcept override { return what_.c_str(); }

  uint8_t exit_code = EX_SOFTWARE;  // if error should cause exit, this should be used.
 protected:
  mutable std::string what_;
};
}  // namespace lox

#endif  // LOX_INCLUDES_LOX_LOX_ERROR_H_
