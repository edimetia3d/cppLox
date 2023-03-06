

cc_library(evaluator
    evaluator.cc
    environment.cc
    runtime_object.cc)

target_link_libraries(lox.backend.tree_walker.evaluator PRIVATE lox.ast)
target_link_libraries(lox.backend.tree_walker.evaluator PRIVATE lox.common)
target_link_libraries(lox.backend.tree_walker.evaluator PRIVATE lox.backend.tree_walker.builtins)
