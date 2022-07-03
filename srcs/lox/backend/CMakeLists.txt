
add_library(backend bakcend.cc)

add_subdirectory(tree_walker)
add_subdirectory(virtual_machine)

target_link_libraries(backend PRIVATE tree_walker virtual_machine)

if (ENABLE_MLIR_JIT_BACKEND)
    add_subdirectory(mlir)
    add_subdirectory(llvm)
    target_link_libraries(backend PRIVATE mlir_backend llvm_backend)
endif ()
