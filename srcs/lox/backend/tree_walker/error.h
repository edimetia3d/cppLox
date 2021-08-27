//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/frontend/token.h"
#include "lox/lox_error.h"

namespace lox {

using RuntimeError = PrefixTokenError<"RuntimeError">;
using ResolveError = PrefixTokenError<"ResolveError">;
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
