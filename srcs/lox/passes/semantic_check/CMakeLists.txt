

cc_library(semantic_check semantic_check.cc)
target_link_libraries(lox.passes.semantic_check PRIVATE lox.ast)
