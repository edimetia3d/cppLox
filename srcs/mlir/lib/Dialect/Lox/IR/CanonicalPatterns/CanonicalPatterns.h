//
// License: MIT
//
#ifndef LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CANONICALPATTERNS_CANONICALPATTERNS_H
#define LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CANONICALPATTERNS_CANONICALPATTERNS_H
#include "llvm/Support/Compiler.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::lox {
void populateReshapeCanonicalPatterns(::mlir::RewritePatternSet &patterns);
void populateTransposeCanonicalPatterns(::mlir::RewritePatternSet &patterns);
} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CANONICALPATTERNS_CANONICALPATTERNS_H