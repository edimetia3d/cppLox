//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H

#include "lox/lox_error.h"
namespace lox::vm {
class CompilationError : public LoxError {
  using LoxError::LoxError;
};

class RuntimeError : public LoxError {
  using LoxError::LoxError;
};
}  // namespace lox::vm

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H
