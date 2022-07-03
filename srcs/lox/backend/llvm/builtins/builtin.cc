//
// LICENSE: MIT
//

#include "lox/backend/llvm/builtins/builtin.h"

#include <cstdio>

void __lox_jit_println_num(double value) { printf("%g\n", value); }
void __lox_jit_println_nil() { printf("nil\n"); }
void __lox_jit_println_str(int8_t *arg) { printf("%s\n", (char *)arg); }
void __lox_jit_println_bool(int8_t value) { printf("%s\n", value ? "true" : "false"); }
