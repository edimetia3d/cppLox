//
// License: MIT
//

#ifndef LOX_AST_TO_LLVM_H
#define LOX_AST_TO_LLVM_H

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "lox/ast/ast.h"

namespace lox::llvm_jit {

std::unique_ptr<llvm::Module> ConvertASTToLLVM(llvm::LLVMContext &context, lox::Module *root);
}  // namespace lox::llvm_jit

#endif  // LOX_AST_TO_LLVM_H
