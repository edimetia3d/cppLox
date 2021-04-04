//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_
#include <memory>
#include <string>
#include <vector>

namespace lox {
/**
 * Note: only class Lox will handle Errors, all other class only generate
 * errors, and return `Error` to the caller.
 */
class Error {
 public:
  Error();
  explicit Error(const std::string &message);
  std::string Message();
  int ToErrCode();
  void Append(const Error &new_err);

 private:
  std::string message_;
  std::shared_ptr<Error> next_;
  std::shared_ptr<Error> tail_;
};

#define ERR_STR(STR)                                     \
  Error(std::string("[") + std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + "] " + STR)
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
