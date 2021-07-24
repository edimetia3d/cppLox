//
// License: MIT
//

#include "lox/error.h"

namespace lox {
int Error::ToErrCode() { return Message().size(); }

void Error::Append(const Error &new_err) {
  ErrorNode node = std::make_shared<Error>(new_err);
  sub_errors.push_back(node);
}
std::string Error::Message() { return RecursiveMessage(0); }
Error::Error() {}

Error::Error(const std::string &message) : message_(message) {}

std::string Error::RecursiveMessage(int level) {
  std::string tab_base = "  ";
  std::string tab = "";
  for (int i = 0; i < level; ++i) {
    tab += tab_base;
  }
  auto ret = tab + message_;
  for (auto node : sub_errors) {
    ret += std::string("\n");
    ret += node->RecursiveMessage(level + 1);
  }
  return ret;
}
Error::Error(const Token &token, const std::string &message) {
  message_ =
      std::to_string(token.line_) + " " + token.lexeme_ + ":" + " " + message;
}
}  // namespace lox
