//
// License: MIT
//

/**
 * The upstream toy tutorial uses tablegen go generate patterns in this file. However, we use c++ instead, for reasons
 * below:
 * 1. Downstream user like us does not have so many dialect/pattern to maintain, so write tablegen based pattern will
 * not bring too much speedup.
 * 2. Unlike Operation/Type, the MLIR framework itself is highly decoupled from the Patterns, Handwrite a c++ pattern
 * correctly is much easier than handwrite a correct c++ operation/type.
 * 3. You will always need to write some C++ pattern someday. So, give your brain a break, by learning less things.
 */

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

#include "CanonicalPatterns.h"

namespace mlir::lox {
namespace {
struct ReshapeReshape : public OpRewritePattern<ReshapeOp> {

  ReshapeReshape(MLIRContext *context) : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
    // Look through the input of the current reshape.
    Value opd0 = op.getInput();
    ReshapeOp opd0_src = opd0.getDefiningOp<ReshapeOp>();

    // Input defined by another reshape? If not, no match.
    if (!opd0_src)
      return failure();

    // Replace current op's operand 0 with reshapeInputOp's oprand 0 inplace
    rewriter.updateRootInPlace(op, [&]() { op.setOperand(opd0_src.getOperand()); });

    return success();
  }
};

struct ReshapeConstant : public OpRewritePattern<ReshapeOp> {
  ReshapeConstant(MLIRContext *context) : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
    // Look through the input of the current reshape.
    Value opd0 = op.getOperand();
    ConstantOp opd0_src = opd0.getDefiningOp<ConstantOp>();

    // Input defined by constant? If not, no match.
    if (!opd0_src)
      return failure();

    // Replace current op with a new constant op
    auto old_dense = opd0_src.getValue().dyn_cast<mlir::DenseElementsAttr>();
    assert(old_dense);
    const auto &result_t = op.getResult().getType();
    auto new_dense = old_dense.reshape(result_t.cast<mlir::ShapedType>());
    auto new_constant = rewriter.create<ConstantOp>(op.getLoc(), result_t, new_dense);
    rewriter.replaceOp(op, {new_constant.getResult()});
    return success();
  }
};

struct EraseReshapeIdentical : public OpRewritePattern<ReshapeOp> {
  EraseReshapeIdentical(MLIRContext *context) : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
    if (op.getOperand().getType() != op.getResult().getType()) {
      return failure();
    }
    // replace all uses and erase current op
    rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void LLVM_ATTRIBUTE_UNUSED populateReshapeCanonicalPatterns(RewritePatternSet &patterns) {
  patterns.add<ReshapeReshape, ReshapeConstant, EraseReshapeIdentical>(patterns.getContext());
}
} // namespace mlir::lox