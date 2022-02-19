
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <numeric>

#include "mlir/Dialect/lox/Dialect.h"
using namespace mlir;
using namespace mlir::lox;

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context) : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp) return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
