include(third_party_llvm)
setup_llvm()

# fixme: update source code so that -Wl,-z,defs could be applied
message(DEBUG "Linker flags: ${CMAKE_SHARED_LINKER_FLAGS} changed to \"-Wl,-z,nodelete\"")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,nodelete")

add_subdirectory(translation)

cc_library(mlir SHARED mlir_jit.cpp)

target_link_libraries(lox.backend.mlir
    PRIVATE
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRLoxAll
    lox.common
    lox.backend.mlir.translation
    )
llvm_update_compile_flags(lox.backend.mlir)
mlir_check_all_link_libraries(lox.backend.mlir)