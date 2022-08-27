//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_RESHAPEREWRITER_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_RESHAPEREWRITER_H

#include <llvm/Support/Compiler.h>

#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::lox {
void LLVM_ATTRIBUTE_UNUSED populateReshapeCanonicalRewriter(::mlir::RewritePatternSet &patterns);
}
#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_RESHAPEREWRITER_H
