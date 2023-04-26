//
// License: MIT
//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "mlir/Conversion/LoxToMixedLox/LoxToMixedLox.h"
#include "mlir/Dialect/Lox/IR/LoxDialect.h"

using namespace mlir;

namespace {
/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands, PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  buildAffineLoopNest(rewriter, loc, lowerBounds, tensorType.getShape(), steps,
                      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                        // Call the processing function with the rewriter, the memref operands,
                        // and the loop induction variables. This function will return the value
                        // to store at the current index.
                        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
                        nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
                      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp> struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
      // Generate an adaptor for the remapped operands of the
      // BinaryOp. This allows for using the nice named accessors
      // that are generated by the ODS.
      typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

      // Generate loads for the element of 'lhs' and 'rhs' at the
      // inner loop.
      auto loadedLhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.getLhs(), loopIvs);
      auto loadedRhs = builder.create<AffineLoadOp>(loc, binaryAdaptor.getRhs(), loopIvs);

      // Create the binary operation performed on the loaded
      // values.
      return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
    });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<lox::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<lox::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<lox::ConstantOp> {
  using OpRewritePattern<lox::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lox::ConstantOp op, PatternRewriter &rewriter) const final {
    if (!op.getValue().getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, [](Diagnostic &diag) { diag << "only 'lox.constant' that produce tensor could be matched"; });
    }
    DenseElementsAttr constantValue = op.getValue().cast<DenseElementsAttr>();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                                       llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<lox::FuncOp> {
  using OpConversionPattern<lox::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(lox::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(
          op, [](Diagnostic &diag) { diag << "expected 'main' to have 0 inputs and 0 results"; });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<lox::PrintOp> {
  using OpConversionPattern<lox::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(lox::PrintOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op, [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<lox::ReturnOp> {
  using OpRewritePattern<lox::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lox::ReturnOp op, PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx) : ConversionPattern(lox::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
      // Generate an adaptor for the remapped operands of the
      // TransposeOp. This allows for using the nice named
      // accessors that are generated by the ODS.
      lox::TransposeOpAdaptor transposeAdaptor(memRefOperands);
      Value input = transposeAdaptor.getInput();

      // Transpose the elements by generating a load from the
      // reverse indices.
      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
      return builder.create<AffineLoadOp>(loc, input, reverseIvs);
    });
    return success();
  }
};

} // namespace

void mlir::lox::populateLoxToMixedLoxPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering, PrintOpLowering, ReturnOpLowering,
               TransposeOpLowering>(patterns.getContext());
}