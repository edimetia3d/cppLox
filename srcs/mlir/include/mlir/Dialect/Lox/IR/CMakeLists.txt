add_mlir_dialect(Lox lox)

set(LLVM_TARGET_DEFINITIONS LoxBase.td)
mlir_tablegen(LoxInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(LoxInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLoxInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRLoxInterfaceIncGen)