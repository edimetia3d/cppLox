//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H

#include "lox/lox_error.h"
namespace lox::vm {
class CompilationError : public LoxErrorWithExitCode<EX_DATAERR> {
  using LoxErrorWithExitCode<EX_DATAERR>::LoxErrorWithExitCode;
};

class RuntimeError : public LoxErrorWithExitCode<EX_SOFTWARE> {
  using LoxErrorWithExitCode<EX_SOFTWARE>::LoxErrorWithExitCode;
};
}  // namespace lox::vm

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_ERRORS_H
