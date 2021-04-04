//
// License: MIT
//

#include "lox/error.h"

namespace lox {
int Error::TOErrCode() { return 0; }

void Error::Append(const Error &new_err) {}
const std::string &Error::Message() { return message_; }
Error::Error() {}
Error::Error(const std::string &message) : message_(message) {}
}  // namespace lox