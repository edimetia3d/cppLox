//
// License: MIT
//

#ifndef LOX_AST_TO_LLVM_H
#define LOX_AST_TO_LLVM_H

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "lox/ast/ast.h"
#include "lox/backend/llvm/builtins/builtin.h"

namespace lox::llvm_jit {

struct ConvertedModule {
  std::unique_ptr<llvm::Module> def_module;
  std::unique_ptr<llvm::Module>
      init_module; // contains global expression that will be evaluated after def_module is loaded
};

ConvertedModule ConvertASTToLLVM(llvm::LLVMContext &context, lox::Module *root, KnownGlobalSymbol *known_global_symbol);
} // namespace lox::llvm_jit

#endif // LOX_AST_TO_LLVM_H
