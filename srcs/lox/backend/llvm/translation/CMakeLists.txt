

llvm_map_components_to_libnames(llvm_libs support core)
cc_library(translation ast_to_llvm.cpp)
target_link_libraries(lox.backend.llvm.translation PUBLIC ${llvm_libs} lox.common)
set_target_properties(lox.backend.llvm.translation PROPERTIES OUTPUT_NAME "llvm_translation")
