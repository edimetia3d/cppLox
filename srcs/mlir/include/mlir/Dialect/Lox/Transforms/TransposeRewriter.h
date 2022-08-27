//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_TRANSPOSEREWRITER_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_TRANSPOSEREWRITER_H

#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::lox {
void LLVM_ATTRIBUTE_UNUSED populateTransposeCanonicalRewriter(::mlir::RewritePatternSet &patterns);
} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_TRANSFORMS_TRANSPOSEREWRITER_H
