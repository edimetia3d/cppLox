//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_
#define LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_
#include <stdint.h>

extern "C" {
void __lox_jit_println_num(double value);
void __lox_jit_println_nil();
void __lox_jit_println_str(int8_t* arg);
void __lox_jit_println_bool(int8_t value);
}

#endif  // LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_
