//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BULTINS_BUILTIN_FN_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BULTINS_BUILTIN_FN_H

#include <map>
#include <string>

#include "lox/object/value.h"

/**
 * Every native function is a `Value (int argCount, Value* args)` style
 * function pointer.
 * The args[0],args[1]... will be the corresponding argument `foo(arg0, arg1...)`in the source code.
 * The return value will be leave on stack by the vm (which is the caller of native function).
 *
 * Note that:
 * `args` is a pointer to stack, Native function are free to access anything on stack
 * But there is no reason to do that for now.
 */

namespace lox::vm {
using NativeFn = Value (*)(int argCount, Value* args);
const std::map<std::string, NativeFn>& AllNativeFn();

}  // namespace lox::vm
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_BULTINS_BUILTIN_FN_H
