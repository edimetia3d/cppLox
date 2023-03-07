include(HandleLLVMOptions)
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core)
add_llvm_library(lox.backend.llvm.translation ast_to_llvm.cpp)
target_link_libraries(lox.backend.llvm.translation PRIVATE ${llvm_libs})
target_link_libraries(lox.backend.llvm.translation PRIVATE lox.common)
set_target_properties(lox.backend.llvm.translation PROPERTIES OUTPUT_NAME "llvm_translation")
