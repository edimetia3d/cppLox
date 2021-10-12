//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_BUILTIN_FN_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_BUILTIN_FN_H

#include "lox/backend/virtual_machine/common/clox_value.h"

namespace lox {
namespace vm {
Value clockNative(int argCount, Value* args);
}
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_BUILTIN_FN_H
