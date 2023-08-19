
cc_binary(lox-bin main.cc)
target_link_libraries(bin.lox-bin lox)
set_target_properties(bin.lox-bin PROPERTIES OUTPUT_NAME "lox")

cc_binary(lox-format lox-format.cc)
target_link_libraries(bin.lox-format lox.frontend lox.passes.ast_printer)
