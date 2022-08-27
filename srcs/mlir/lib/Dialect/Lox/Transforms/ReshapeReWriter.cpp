//
// License: MIT
//
#include "mlir/Dialect/Lox/Transforms/ReshapeReWriter.h"
#include "mlir/Dialect/Lox/IR/Lox.h"
using namespace mlir;
namespace {
#include "mlir/Dialect/Lox/Transforms/ReshapeRewriter.cpp.inc"
}
namespace mlir::lox {
void LLVM_ATTRIBUTE_UNUSED populateReshapeCanonicalRewriter(::mlir::RewritePatternSet &patterns) {
  return populateWithGenerated(patterns);
}
} // namespace mlir::lox