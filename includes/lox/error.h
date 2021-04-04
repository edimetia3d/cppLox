//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_
#include <string>

namespace lox {
class Error {
 public:
  Error();
  explicit Error(const std::string &message);
  const std::string &Message();
  int TOErrCode();
  void Append(const Error &new_err);

 private:
  std::string message_;
};

#define ERR_STR(STR)                                     \
  Error(std::string("[") + std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + "] " + STR)
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
