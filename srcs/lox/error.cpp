//
// License: MIT
//

#include "lox/error.h"

namespace lox {
int Error::ToErrCode() { return Message().size(); }

void Error::Append(const Error &new_err) {
  if (next_) {
    tail_->next_ = std::make_shared<Error>(new_err);
    tail_ = tail_->next_;
  } else {
    next_ = std::make_shared<Error>(new_err);
    tail_ = next_;
  }
}
std::string Error::Message() {
  std::string ret = message_;
  auto next = next_;
  while (next) {
    ret += (next->message_ + "\n");
    next = next->next_;
  }
  return ret;
}
Error::Error() {}
Error::Error(const std::string &message) : message_(message) {}
}  // namespace lox