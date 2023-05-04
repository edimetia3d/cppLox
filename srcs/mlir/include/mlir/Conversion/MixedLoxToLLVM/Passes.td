/**
 * Use tablegen to gen a base class for pass to help pass registration
 */

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_PASS_TD
#define LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_PASS_TD



include "mlir/Pass/PassBase.td"

def MixedLoxToLLVM : Pass<"mixedlox-to-llvm", "::mlir::ModuleOp"> {
  let summary = "The conversion to llvm";
  let description = [{
    This pass converts the MixedLox dialect to LLVM dialect
  }];

  // The cpp code used to create pass
  let constructor = "mlir::lox::createLowerMixedLoxToLLVMPass()";
}

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_CONVERSION_MIXEDLOXTOLLVM_PASS_TD
