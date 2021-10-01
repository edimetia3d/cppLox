//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_ERROR_H_
#define CPPLOX_INCLUDES_LOX_ERROR_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/lox_error.h"
#include "lox/token/token.h"

namespace lox {

class RuntimeError : public PrefixTokenError<RuntimeError> {
 public:
  using PrefixTokenError<RuntimeError>::PrefixTokenError;
  static std::string StrName() { return "RuntimeError"; }
};
class ResolveError : public PrefixTokenError<ResolveError> {
 public:
  using PrefixTokenError<ResolveError>::PrefixTokenError;
  static std::string StrName() { return "ResolveError"; }
};
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_ERROR_H_
