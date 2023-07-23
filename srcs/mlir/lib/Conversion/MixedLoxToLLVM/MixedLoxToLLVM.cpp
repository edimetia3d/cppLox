//
// License: MIT
//

//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/Sequence.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir/Conversion/MixedLoxToLLVM/MixedLoxToLLVM.h"

namespace mlir::lox {

#define GEN_PASS_CLASSES
#include "mlir/Conversion/MixedLoxToLLVM/MixedLoxToLLVMBase.h.inc"

namespace {
struct MixedLoxToLLVMPass : public MixedLoxToLLVMBase<MixedLoxToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MixedLoxToLLVMPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  lox::populateMixedLoxToLLVMPatterns(patterns);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerMixedLoxToLLVMPass() { return std::make_unique<MixedLoxToLLVMPass>(); }
} // namespace mlir::lox