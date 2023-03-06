
cc_library(backend bakcend.cc)

add_subdirectory(tree_walker)
add_subdirectory(virtual_machine)

target_link_libraries(lox.backend PRIVATE
    lox.backend.tree_walker
    lox.backend.virtual_machine
    lox.passes.ast_printer
    lox.passes.semantic_check
    lox.passes.pass_runner)

if (ENABLE_JIT_BACKEND)
  add_subdirectory(mlir)
  add_subdirectory(llvm)
  target_link_libraries(lox.backend PRIVATE lox.backend.mlir lox.backend.llvm)
endif ()
