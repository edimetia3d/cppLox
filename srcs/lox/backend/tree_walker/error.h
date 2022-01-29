//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/common/lox_error.h"
#include "lox/token/token.h"

namespace lox {

class RuntimeError : public ErrorWithToken<RuntimeError> {
 public:
  using ErrorWithToken<RuntimeError>::ErrorWithToken;
  static std::string StrName() { return "RuntimeError"; }
};
class ResolveError : public ErrorWithToken<ResolveError> {
 public:
  using ErrorWithToken<ResolveError>::ErrorWithToken;
  static std::string StrName() { return "ResolveError"; }
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
