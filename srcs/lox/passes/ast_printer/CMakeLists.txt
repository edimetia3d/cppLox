
cc_library(ast_printer ast_printer.cc)
target_link_libraries(lox.passes.ast_printer PRIVATE lox.ast)