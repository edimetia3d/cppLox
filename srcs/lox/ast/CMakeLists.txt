
add_library(ast
        ast_printer.cpp
        )

find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_custom_target(generated_expr_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc Expr
            COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc ${CMAKE_CURRENT_LIST_DIR}/expr_decl.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    target_compile_definitions(ast PUBLIC DYNAMIC_GEN_EXPR_DECL)
    add_dependencies(ast generated_expr_decl)
else ()
    message(WARNING "Using cached expr_decl, it might be out of sync!")
endif ()
