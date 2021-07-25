
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc
        COMMAND python3 nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
add_library(ast
        expr.h
        ${CMAKE_CURRENT_BINARY_DIR}/expr_decl.h.inc
        ast_printer.cpp
        )
