
add_library(expr_stmt
        INTERFACE
        )

add_library(ast_printer
        ast_printer.cpp
        )

add_library(evaluator
        eval_visitor.cpp
        )

target_link_libraries(evaluator PUBLIC lox_object)
target_link_libraries(evaluator PRIVATE ast_printer)
target_link_libraries(evaluator PUBLIC expr_stmt)


find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    target_compile_definitions(expr_stmt INTERFACE DYNAMIC_GEN_DECL)

    add_custom_target(generated_expr_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc Expr
            COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc ${CMAKE_CURRENT_LIST_DIR}/expr_decl.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(expr_stmt generated_expr_decl)

    add_custom_target(generated_stmt_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc Stmt
            COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc ${CMAKE_CURRENT_LIST_DIR}/stmt_decl.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(expr_stmt generated_stmt_decl)

else ()
    message(WARNING "Using cached expr_decl, it might be out of sync!")
endif ()
