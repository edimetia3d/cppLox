//
// License: MIT
//

#ifndef LOX_AST_TO_MLIR_H
#define LOX_AST_TO_MLIR_H

#include <mlir/IR/BuiltinOps.h>

#include "lox/ast/ast.h"

namespace lox::jit {

mlir::OwningModuleRef ConvertASTToMLIR(mlir::MLIRContext &context, lox::Module *lox_module);
}  // namespace lox::jit

#endif  // LOX_AST_TO_MLIR_H
