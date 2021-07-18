
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc
        COMMAND python3 nodes_code_writer.py ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
add_library(expr_decls expr.h ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc)
set_target_properties(expr_decls PROPERTIES LINKER_LANGUAGE CXX)