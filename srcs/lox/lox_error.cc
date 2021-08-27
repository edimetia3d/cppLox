//
// LICENSE: MIT
//
#include "lox/lox_error.h"

namespace lox {
int LoxError::ToErrCode() const { return message_.size() + sub_errors.size(); }

void LoxError::Merge(const LoxError &new_err) {
  if (new_err.ToErrCode()) {
    if (message_.empty()) {
      *this = new_err;
    } else {
      ErrorNode node = std::make_shared<LoxError>(new_err);
      sub_errors.push_back(node);
    }
  }
}
std::string LoxError::Message() const { return RecursiveMessage(0); }
LoxError::LoxError() {}

LoxError::LoxError(const std::string &message) : message_(message) {}

std::string LoxError::RecursiveMessage(int level) const {
  std::string ret;
  if (!message_.empty()) {
    std::string tab_base = "  ";
    std::string tab = "";
    for (int i = 0; i < level; ++i) {
      tab += tab_base;
    }
    ret = tab + message_;
    for (auto node : sub_errors) {
      ret += std::string("\n");
      ret += node->RecursiveMessage(level + 1);
    }
  }
  return ret;
}
}  // namespace lox
