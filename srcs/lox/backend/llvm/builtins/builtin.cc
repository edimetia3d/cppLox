//
// LICENSE: MIT
//

#include "lox/backend/llvm/builtins/builtin.h"

#include <llvm/IR/Function.h>

#include <cstdio>

extern "C" {
void __lox_jit_println_num(double value) { printf("%g\n", value); }
void __lox_jit_println_nil() { printf("nil\n"); }
void __lox_jit_println_str(int8_t *arg) { printf("%s\n", (char *)arg); }
void __lox_jit_println_bool(int8_t value) { printf("%s\n", value ? "true" : "false"); }
}
void lox::llvm_jit::RegBuiltin(llvm::LLVMContext &context, lox::llvm_jit::KnownGlobalSymbol *known_global_symbol) {
  auto num_ty = llvm::Type::getDoubleTy(context);
  auto str_ty = llvm::Type::getInt8PtrTy(context);
  auto nil_ty = llvm::Type::getVoidTy(context);
  auto i8_ty = llvm::Type::getInt8Ty(context);

  auto print_num_fn_ty = llvm::FunctionType::get(nil_ty, {num_ty}, false);
  auto print_str_fn_ty = llvm::FunctionType::get(nil_ty, {str_ty}, false);
  auto print_bool_fn_ty = llvm::FunctionType::get(nil_ty, {i8_ty}, false);
  auto print_nil_fn_ty = llvm::FunctionType::get(nil_ty, {}, false);

  known_global_symbol->insert(std::make_pair("__lox_jit_println_num", print_num_fn_ty));
  known_global_symbol->insert(std::make_pair("__lox_jit_println_str", print_str_fn_ty));
  known_global_symbol->insert(std::make_pair("__lox_jit_println_bool", print_bool_fn_ty));
  known_global_symbol->insert(std::make_pair("__lox_jit_println_nil", print_nil_fn_ty));
}
