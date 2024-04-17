//
// LICENSE: MIT
//
#include "lox/common/lox_error.h"

namespace lox {

const char *LoxError::what() const noexcept { return what_.c_str(); }
} // namespace lox
