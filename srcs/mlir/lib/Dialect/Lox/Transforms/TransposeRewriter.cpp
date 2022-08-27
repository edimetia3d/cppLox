
#include "mlir/Dialect/Lox/Transforms/TransposeRewriter.h"

#include "mlir/Dialect/Lox/IR/Lox.h"

using namespace mlir;
using namespace mlir::lox;
namespace {
struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(MLIRContext *context) : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

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

namespace mlir::lox {
void LLVM_ATTRIBUTE_UNUSED populateTransposeCanonicalRewriter(::RewritePatternSet &patterns) {
  patterns.add<SimplifyRedundantTranspose>(patterns.getContext());
}
} // namespace lox