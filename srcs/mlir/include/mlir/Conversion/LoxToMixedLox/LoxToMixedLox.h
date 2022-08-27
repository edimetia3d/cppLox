//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_LOXTOMIXEDLOX_LOXTOMIXEDLOX_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_LOXTOMIXEDLOX_LOXTOMIXEDLOX_H
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
namespace mlir::lox {
void populateLoxToMixedLoxPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createLowerLoxToMixedLoxPass();
} // namespace mlir::lox
#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_LOXTOMIXEDLOX_LOXTOMIXEDLOX_H
