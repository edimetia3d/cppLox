//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_
#define LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <stdint.h>

#include <unordered_map>

namespace lox::llvm_jit {
using KnownGlobalSymbol = std::unordered_map<std::string, llvm::Type *>;

void RegBuiltin(llvm::LLVMContext &context, KnownGlobalSymbol *known_global_symbol);
}  // namespace lox::llvm_jit

#endif  // LOX_SRCS_LOX_BACKEND_LLVM_BUILTINS_BUILTIN_H_
