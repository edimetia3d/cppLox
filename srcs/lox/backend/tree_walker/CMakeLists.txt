
cc_library(tree_walker tree_walker.cc)

add_subdirectory(builtins)
add_subdirectory(evaluator)

target_link_libraries(lox.backend.tree_walker PRIVATE lox.backend.tree_walker.evaluator)
