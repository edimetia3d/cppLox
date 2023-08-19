
#include "CanonicalPatterns.h"

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

namespace mlir::lox {
namespace {
/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(MLIRContext *context) : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override;
};
LogicalResult SimplifyRedundantTranspose::matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const {
  // Look through the input of the current transpose.
  Value transposeInput = op.getOperand();
  TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

  // Input defined by another transpose? If not, no match.
  if (!transposeInputOp)
    return failure();

  // Otherwise, we have a redundant transpose. Use the rewriter.
  rewriter.replaceOp(op, {transposeInputOp.getOperand()});
  return success();
}
} // namespace

void populateTransposeCanonicalPatterns(RewritePatternSet &patterns) {
  patterns.add<SimplifyRedundantTranspose>(patterns.getContext());
}
} // namespace mlir::lox