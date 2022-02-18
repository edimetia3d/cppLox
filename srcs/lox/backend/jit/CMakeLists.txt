
add_subdirectory(translation)

add_library(jit_backend mlir_jit.cpp)
target_link_libraries(jit_backend ast_to_mlir)