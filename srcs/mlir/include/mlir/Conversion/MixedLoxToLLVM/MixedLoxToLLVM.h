//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_MIXEDLOXTOLLVM_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_MIXEDLOXTOLLVM_H
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace mlir::lox {
void populateMixedLoxToLLVMPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createLowerMixedLoxToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/MixedLoxToLLVM/MixedLoxToLLVMBase.h.inc"

} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_MIXEDLOXTOLLVM_H
