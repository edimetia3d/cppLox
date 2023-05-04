set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(MixedLoxToLLVMBase.h.inc -gen-pass-decls -name MixedLoxToLLVM)
add_public_tablegen_target(MLIRMixedLoxToLLVMIncGen)